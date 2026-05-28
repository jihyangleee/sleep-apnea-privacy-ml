import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyActivation(nn.Module):
    """HE-compatible polynomial activation: f(x) = x² + 0.5x.

    Factored as x * (x + 0.5) so that HE inference needs only one
    ciphertext multiplication (one level consumed) rather than two.
    """
    def forward(self, x):
        return x * (x + 0.5)


class ClientSubModel(nn.Module):
    """Vertical FL client sub-model: feature columns → embedding."""
    def __init__(self, input_dim: int, emb_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.act    = PolyActivation()

    def forward(self, x):
        return self.act(self.linear(x))


class ServerTopModel(nn.Module):
    """Vertical FL top-model: concatenated embeddings → logit.

    Structure: Linear1 → PolyAct → Linear2 → logit
      - PolyAct captures cross-hospital feature interactions in the hidden layer.
      - sigmoid is NOT applied here; BCEWithLogitsLoss handles it during training,
        and the patient applies sigmoid locally after decrypting in inference.
    """
    def __init__(self, total_emb_dim: int, hidden: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(total_emb_dim, hidden)
        self.act     = PolyActivation()
        self.linear2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class VerticalHeartNet(nn.Module):
    """Full Vertical FL model for HE inference — sub-models + top-model combined."""
    def __init__(self, feature_groups: list, emb_dim: int = 16):
        super().__init__()
        self.feature_groups = [list(fg) for fg in feature_groups]
        self.emb_dim        = emb_dim
        self.sub_models     = nn.ModuleList([
            ClientSubModel(len(fg), emb_dim) for fg in self.feature_groups
        ])
        self.top_model = ServerTopModel(emb_dim * len(self.feature_groups))

    def forward(self, x):
        embs = [sub(x[:, fg]) for sub, fg in zip(self.sub_models, self.feature_groups)]
        return self.top_model(torch.cat(embs, dim=1))


class HospitalModel(nn.Module):
    """Per-hospital model: private sub-model + distributed top-model.

    Training  — secret shares + Beaver Triple (no central server sees embeddings)
    Inference — CKKS ciphertext arithmetic (patient features never decrypted)

    The top-model weights (top_W1, top_W2) are logically identical across all
    hospitals. In simulation, every HospitalModel receives references to the
    same nn.Linear objects. In production they would be synchronised via
    AllReduce / FedAvg gradient aggregation.
    """

    def __init__(
        self,
        hospital_id: int,
        feature_groups: list,           # all hospitals' feature-column lists
        emb_dim: int = 16,
        top_hidden: int = 32,
        shared_top_W1: "nn.Linear | None" = None,
        shared_top_W2: "nn.Linear | None" = None,
    ):
        super().__init__()
        self.id            = hospital_id
        self.feature_groups = [list(fg) for fg in feature_groups]
        self.emb_dim       = emb_dim
        self.n             = len(feature_groups)
        self.n_features    = sum(len(fg) for fg in feature_groups)
        self.total_emb     = emb_dim * self.n

        # Sub-model: private to this hospital, processes own feature columns only
        self.sub = ClientSubModel(len(feature_groups[hospital_id]), emb_dim)

        # Top-model: shared weights (same nn.Linear instance across all hospitals)
        if shared_top_W1 is None:
            self.top_W1 = nn.Linear(self.total_emb, top_hidden)
            self.top_W2 = nn.Linear(top_hidden, 1)
        else:
            self.top_W1 = shared_top_W1
            self.top_W2 = shared_top_W2

        self._he_built = False

    # ── Training helpers ──────────────────────────────────────────────────────

    def local_emb(self, x_local: torch.Tensor) -> torch.Tensor:
        """Sub-model forward on own features. Plaintext, stays local."""
        return self.sub(x_local)

    def h_share(self, cat_share: torch.Tensor, is_first: bool) -> torch.Tensor:
        """Apply W1 (no bias) to a received share. Only party 0 adds bias.

        sum_j( h_share_j ) = full_cat @ W1.T + W1.bias = h_linear
        """
        h = F.linear(cat_share, self.top_W1.weight)
        if is_first:
            h = h + self.top_W1.bias
        return h

    def logit_share(self, act_share: torch.Tensor, is_first: bool) -> torch.Tensor:
        """Apply W2 (no bias) to an activation share. Only party 0 adds bias."""
        logit = F.linear(act_share, self.top_W2.weight)
        if is_first:
            logit = logit + self.top_W2.bias
        return logit

    # ── Inference helpers (CKKS) ──────────────────────────────────────────────

    def build_he_weights(
        self,
        W1_T_share,   # np.ndarray (total_emb, top_hidden): additive share of W1.T
        b1_share,     # np.ndarray (top_hidden,): additive share of b1
        W2_T_share,   # np.ndarray (top_hidden, 1): additive share of W2.T
        b2_share,     # np.ndarray (1,): additive share of b2
    ):
        """Store sub-model weights and pre-split top-model shares for CKKS inference.

        Sub-model: hospital i receives only enc(features_i) from the watch.
          W_sub_T (|fg_i|, emb_dim) applied directly — no zero-padding needed.

        Top-model: W1 and W2 are additively split across hospitals.
          W1_T_share (total_emb, top_hidden) is stored as n row-blocks of
          (emb_dim, top_hidden), one block per source embedding.
          This lets each hospital compute its W1_share contribution from
          the individual enc(emb_j) ciphertexts without forming enc(concat).

          math: enc(emb_j) @ W1_T_share[j*d:(j+1)*d, :]  for each j
                sum over j  =  enc(concat) @ W1_T_share  =  enc(h_linear share_i)
                sum over i  =  enc(h_linear)  ✓
        """
        import numpy as np

        W_sub = self.sub.linear.weight.detach().numpy()   # (emb_dim, |fg_i|)
        b_sub = self.sub.linear.bias.detach().numpy()     # (emb_dim,)
        self._W_sub_T = W_sub.T.astype(np.float64).tolist()   # (|fg_i|, emb_dim)
        self._b_sub   = b_sub.astype(np.float64).tolist()     # (emb_dim,)

        # Pre-slice W1_T_share into n blocks, one per source embedding
        W1_arr = W1_T_share if isinstance(W1_T_share, np.ndarray) else np.array(W1_T_share)
        self._W1_T_share_blocks = [
            W1_arr[j * self.emb_dim : (j + 1) * self.emb_dim].tolist()
            for j in range(self.n)
        ]
        self._b1_share = b1_share.tolist() if hasattr(b1_share, "tolist") else b1_share

        W2_arr = W2_T_share if isinstance(W2_T_share, np.ndarray) else np.array(W2_T_share)
        self._W2_T_share = W2_arr.tolist()
        self._b2_share   = b2_share.tolist() if hasattr(b2_share, "tolist") else b2_share

        self._he_built = True

    def compute_sub_emb_he(self, enc_xi_bytes: bytes, ctx) -> bytes:
        """CKKS sub-model: enc(features_i) -> enc(emb_i).

        Receives only this hospital's feature slice (size |fg_i|) from the watch.
        No zero-padding: W_sub_T maps (|fg_i|,) -> (emb_dim,) directly.
        PolyAct applied in ciphertext space (-1 HE level).
        """
        import tenseal as ts
        assert self._he_built, "call build_he_weights() after training"

        enc_h = ts.lazy_ckks_vector_from(enc_xi_bytes)
        enc_h.link_context(ctx)
        enc_h.mm_(self._W_sub_T)    # (|fg_i|,) -> (emb_dim,)
        enc_h += self._b_sub
        enc_act = enc_h * (enc_h + 0.5)   # PolyAct (-1 HE level)
        return enc_act.serialize()

    def compute_h_share_he(self, enc_emb_dict: dict, ctx) -> bytes:
        """Apply own W1_share to all enc(emb_j) and sum -> enc(h_share_i).

        enc_emb_dict: {hospital_id: enc_emb_bytes}  (one emb_dim ciphertext each)

        For each source hospital j:
          contribution_j = enc(emb_j) @ W1_T_share_blocks[j]  (emb_dim -> top_hidden)
        enc(h_share_i) = sum_j(contribution_j) + b1_share_i

        Summing across all hospitals i:
          sum_i enc(h_share_i) = enc(concat @ W1.T + b1) = enc(h_linear)  ✓
        No party ever holds or sees enc(concat) as a single ciphertext.
        """
        import tenseal as ts
        assert self._he_built

        enc_h = None
        for j in range(self.n):
            enc_emb_j = ts.lazy_ckks_vector_from(enc_emb_dict[j])
            enc_emb_j.link_context(ctx)
            contrib = enc_emb_j.mm_(self._W1_T_share_blocks[j])
            enc_h = contrib if enc_h is None else enc_h + contrib

        enc_h += self._b1_share
        return enc_h.serialize()

    def compute_logit_share_he(self, enc_h_act_bytes: bytes, ctx) -> bytes:
        """Apply own W2_share + b2_share to enc(h_act) -> enc(logit_share_i).

        Coordinator broadcasts enc(h_act) after PolyAct; each hospital applies
        its additive share of W2. Summing across hospitals gives enc(logit).
        """
        import tenseal as ts
        assert self._he_built

        enc_logit = ts.lazy_ckks_vector_from(enc_h_act_bytes)
        enc_logit.link_context(ctx)
        enc_logit.mm_(self._W2_T_share)
        enc_logit += self._b2_share
        return enc_logit.serialize()

    def run_he_inference(
        self,
        all_enc_xi: dict,           # {hospital_id: enc_xi_bytes} — each hospital's partial features
        ctx,
        other_hospitals: list,      # list[HospitalModel]
    ) -> bytes:
        """Fully distributed CKKS inference. Any hospital can be the entry point.

        Protocol (no party ever sees plaintext features, embeddings, or logit):

        1. Sub-model: each hospital independently decodes enc(features_i) with
           its private sub-model -> enc(emb_i).  Each party sees only its own
           encrypted feature slice from the watch.

        2. Embedding exchange: hospitals broadcast enc(emb_i) to each other.
           All parties now hold {enc(emb_j) for all j}, all as ciphertexts.

        3. Top Linear1 (W1 additive share): each hospital i computes
             enc(h_share_i) = sum_j( enc(emb_j) @ W1_T_share_i[j*d:(j+1)*d] ) + b1_share_i
           Coordinator collects and sums -> enc(h_linear) = enc(concat @ W1 + b1).

        4. PolyAct: coordinator applies h*(h+0.5) to enc(h_linear) in CKKS
           (-1 HE level).  Coordinator sees only ciphertext.

        5. Top Linear2 (W2 additive share): each hospital i computes
             enc(logit_share_i) = enc(h_act) @ W2_T_share_i + b2_share_i
           Coordinator sums -> enc(logit).

        6. Return enc(logit) to patient who decrypts with secret key.
        """
        import tenseal as ts
        assert self._he_built

        all_hospitals = sorted([self] + other_hospitals, key=lambda h: h.id)

        # ── Step 1: each hospital computes enc(emb_i) ─────────────────────────
        enc_emb_dict = {}
        for h in all_hospitals:
            enc_emb_dict[h.id] = h.compute_sub_emb_he(all_enc_xi[h.id], ctx)

        # ── Step 2+3: each hospital applies W1_share; coordinator sums ────────
        enc_h = None
        for h in all_hospitals:
            enc_h_i = ts.lazy_ckks_vector_from(h.compute_h_share_he(enc_emb_dict, ctx))
            enc_h_i.link_context(ctx)
            enc_h = enc_h_i if enc_h is None else enc_h + enc_h_i

        # ── Step 4: PolyAct by coordinator — cross-hospital feature interaction ─
        # Consistent with training: Linear1 → PolyAct → Linear2 → logit.
        # Coordinator sees only ciphertext; nothing is revealed.
        enc_h_act       = enc_h * (enc_h + 0.5)   # -1 HE level
        enc_h_act_bytes = enc_h_act.serialize()

        # ── Step 5: each hospital applies W2_share; coordinator sums → enc_logit
        enc_logit = None
        for h in all_hospitals:
            enc_logit_i = ts.lazy_ckks_vector_from(h.compute_logit_share_he(enc_h_act_bytes, ctx))
            enc_logit_i.link_context(ctx)
            enc_logit = enc_logit_i if enc_logit is None else enc_logit + enc_logit_i

        # Return enc_logit. Patient decrypts and applies sigmoid locally.
        # sigmoid is NOT applied here in CKKS — consistent with training's
        # BCEWithLogitsLoss which also applies sigmoid outside the model output.
        return enc_logit.serialize()

    @staticmethod
    def encrypt_feature_slice(x_slice, ctx) -> bytes:
        """Patient encrypts one hospital's feature slice before sending."""
        import tenseal as ts
        import numpy as np
        return ts.ckks_vector(ctx, x_slice.astype(np.float64).tolist()).serialize()

    @staticmethod
    def decrypt_result(enc_bytes: bytes, ctx) -> float:
        """Patient decrypts enc_logit → sigmoid → probability.

        The model returns a raw logit (same as training's BCEWithLogitsLoss).
        sigmoid is applied here on the patient device, not in CKKS.
        """
        import tenseal as ts
        import numpy as np
        vec = ts.lazy_ckks_vector_from(enc_bytes)
        vec.link_context(ctx)
        logit = vec.decrypt()[0]
        return float(1.0 / (1.0 + np.exp(-logit)))
