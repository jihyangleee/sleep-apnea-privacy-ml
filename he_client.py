"""
Distributed HE inference across 3 hospitals.

Each hospital holds:
  - Its own sub-model weights  (only for its feature columns)
  - The shared top-model weights  (needed when acting as coordinator)

Any hospital can act as the entry point for a patient request:
  1. Coordinator computes its own enc_emb
  2. Forwards enc_x to the other 2 hospitals -> receives their enc_emb
  3. Aggregates (zero-padding makes element-wise sum == concatenation)
  4. Applies top model -> returns enc_logit to patient

The patient (Galaxy Watch / user's device) holds the CKKS secret key.
Hospitals hold only the public key and model weights.
"""

import numpy as np
import tenseal as ts

from model import VerticalHeartNet


def build_he_context() -> ts.Context:
    # Level budget: sub-model PolyAct (-1) + top-model PolyAct (-1) = 2 levels consumed
    # [60,40,40,40,40,40,60] provides 5 usable levels -> 3 remain after inference.
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    return ctx


class HospitalHE:
    """One hospital's HE inference endpoint.

    Sub-model  : knows only its own feature columns (W_ext zero-padded elsewhere).
    Top-model  : shared weights held by every hospital so any can be coordinator.

    Usage
    -----
    hospitals = [HospitalHE(i, model) for i in range(3)]

    # Any hospital can be the entry point:
    enc_logit = hospitals[0].run_inference(enc_x_bytes, ctx, hospitals[1:])
    enc_logit = hospitals[1].run_inference(enc_x_bytes, ctx, [hospitals[0], hospitals[2]])
    enc_logit = hospitals[2].run_inference(enc_x_bytes, ctx, hospitals[:2])
    """

    def __init__(self, hospital_id: int, model: VerticalHeartNet):
        self.id          = hospital_id
        self.feature_groups = model.feature_groups
        self.emb_dim     = model.emb_dim
        self.n_features  = sum(len(fg) for fg in model.feature_groups)
        self.total_emb   = model.emb_dim * len(model.feature_groups)
        self._build_weights(model)

    def _build_weights(self, model: VerticalHeartNet):
        """Extract own sub-model weights (zero-padded) and top-model weights."""
        sub  = model.sub_models[self.id]
        fg   = self.feature_groups[self.id]
        W    = sub.linear.weight.detach().numpy()   # (emb_dim, |fg|)
        b    = sub.linear.bias.detach().numpy()     # (emb_dim,)

        W_ext = np.zeros((self.n_features, self.total_emb), dtype=np.float64)
        b_ext = np.zeros(self.total_emb, dtype=np.float64)
        start = self.id * self.emb_dim
        for out_idx in range(self.emb_dim):
            for j, fi in enumerate(fg):
                W_ext[fi, start + out_idx] = float(W[out_idx, j])
        b_ext[start : start + self.emb_dim] = b.astype(np.float64)

        self.W_ext = W_ext.tolist()
        self.b_ext = b_ext.tolist()

        # Top-model weights (same at every hospital; needed when acting as coordinator)
        W1 = model.top_model.linear1.weight.detach().numpy()  # (32, total_emb)
        self.W_top1_T = W1.T.astype(np.float64).tolist()      # (total_emb, 32)
        self.b_top1   = model.top_model.linear1.bias.detach().numpy().astype(np.float64).tolist()
        W2 = model.top_model.linear2.weight.detach().numpy()  # (1, 32)
        self.W_top2_T = W2.T.astype(np.float64).tolist()      # (32, 1)
        self.b_top2   = model.top_model.linear2.bias.detach().numpy().astype(np.float64).tolist()

    # ── Sub-model endpoint (called by the coordinator) ────────────────────────

    def compute_sub_emb(self, enc_x_bytes: bytes, ctx: ts.Context) -> bytes:
        """Apply this hospital's sub-model to the encrypted feature vector.

        enc_x (n_features,)  ->  enc_emb_i (total_emb,)
        Non-owned feature slots are zeroed via W_ext padding.
        PolyAct: h * (h + 0.5)  [1 level consumed]
        """
        enc_h = ts.lazy_ckks_vector_from(enc_x_bytes)
        enc_h.link_context(ctx)
        enc_h.mm_(self.W_ext)       # linear: enc_x @ W_ext_i
        enc_h += self.b_ext
        enc_act = enc_h * (enc_h + 0.5)   # PolyAct
        return enc_act.serialize()

    # ── Coordinator endpoint (entry point for patient request) ────────────────

    def run_inference(
        self,
        enc_x_bytes: bytes,
        ctx: ts.Context,
        other_hospitals: list,          # list[HospitalHE]
    ) -> bytes:
        """Full HE inference as coordinator.

        1. Compute own enc_emb_i via sub-model.
        2. Ask each other hospital for their enc_emb_j.
        3. Aggregate: enc_cat = sum(enc_emb_*) -- zero-padding makes this == concat.
        4. Apply top model: Linear -> PolyAct -> Linear -> enc_logit.
        5. Return enc_logit (patient decrypts with secret key).
        """
        # 1. Own sub-model
        enc_cat = ts.lazy_ckks_vector_from(self.compute_sub_emb(enc_x_bytes, ctx))
        enc_cat.link_context(ctx)

        # 2. Other hospitals' sub-models (simulates network call to each hospital)
        for h in other_hospitals:
            enc_emb_j = ts.lazy_ckks_vector_from(h.compute_sub_emb(enc_x_bytes, ctx))
            enc_emb_j.link_context(ctx)
            enc_cat = enc_cat + enc_emb_j

        # 3. Top model: Linear1 → PolyAct → Linear2 → enc_logit
        #    enc_cat = enc(concat_emb)  (= sum of enc_emb_i from all hospitals)
        #    logit_hidden = enc_concat @ W_top1 + b1  (linear, -0 HE levels)
        #    hidden       = PolyAct(logit_hidden)      (non-linear, -1 HE level)
        #    enc_logit    = hidden @ W_top2 + b2       (linear, -0 HE levels)
        enc_cat.mm_(self.W_top1_T)
        enc_cat += self.b_top1
        enc_cat  = enc_cat * (enc_cat + 0.5)   # PolyAct (-1 HE level)
        enc_cat.mm_(self.W_top2_T)
        enc_cat += self.b_top2

        # Patient decrypts enc_logit and applies sigmoid locally.
        return enc_cat.serialize()

    # ── Patient-side helpers ──────────────────────────────────────────────────

    @staticmethod
    def encrypt_input(x: np.ndarray, ctx: ts.Context) -> bytes:
        """Patient encrypts their feature vector before sending."""
        return ts.ckks_vector(ctx, x.astype(np.float64).tolist()).serialize()

    @staticmethod
    def decrypt_result(enc_bytes: bytes, ctx: ts.Context) -> float:
        """Patient decrypts the returned logit -> sigmoid -> probability."""
        vec = ts.lazy_ckks_vector_from(enc_bytes)
        vec.link_context(ctx)
        logit = vec.decrypt()[0]
        return float(1.0 / (1.0 + np.exp(-logit)))


# ── Legacy single-server wrapper (kept for benchmark.py compatibility) ────────

class HEInference:
    """Wraps HospitalHE to mimic the old single-server API.

    Used by benchmark.py. Hospital 0 acts as coordinator by default.
    """

    def __init__(self, model: VerticalHeartNet):
        self.hospitals = [HospitalHE(i, model) for i in range(len(model.feature_groups))]

    def run_he_inference(self, enc_input_bytes: bytes, ctx: ts.Context) -> bytes:
        coord  = self.hospitals[0]
        others = self.hospitals[1:]
        return coord.run_inference(enc_input_bytes, ctx, others)

    @staticmethod
    def encrypt_input(x: np.ndarray, ctx: ts.Context) -> bytes:
        return HospitalHE.encrypt_input(x, ctx)

    @staticmethod
    def decrypt_result(enc_bytes: bytes, ctx: ts.Context) -> float:
        return HospitalHE.decrypt_result(enc_bytes, ctx)
