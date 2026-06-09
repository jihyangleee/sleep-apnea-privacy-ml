import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyActivation(nn.Module):
    """HE-compatible cubic ReLU approximation: f(x) = 0.197x + 0.004x³.

    Factored as x * (0.197 + 0.004*x²) so CKKS inference needs exactly two
    ciphertext multiplications (-2 HE levels), same total as the old
    quadratic sub + quadratic top approach.
    """
    def forward(self, x):
        return 0.197 * x + 0.004 * x ** 3


class ClientSubModel(nn.Module):
    """Vertical FL sub-model: feature columns → embedding.

    Structure: Linear → CubicAct → Linear
    """
    def __init__(self, input_dim: int, emb_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.act    = PolyActivation()
        self.embed  = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.embed(self.act(self.linear(x)))


class ServerTopModel(nn.Module):
    """Vertical FL top-model: concatenated embeddings → logit.

    Single linear layer (no hidden, no activation).
    Column-wise split lets each hospital compute its own partial logit
    from its own embedding without any inter-hospital communication:
      logit = Σ_i (emb_i @ W_top_i.T) + b
    """
    def __init__(self, total_emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(total_emb_dim, 1)

    def forward(self, x):
        return self.linear(x)


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
    """Per-hospital model: private sub-model + shared column slice of top-model.

    Training  — additive SS; linear top-model decomposes over shares (no Beaver Triple needed)
    Inference — CKKS ciphertext arithmetic (patient features never decrypted)

    Inference protocol (3 hospitals, 1 round):
      1. Each hospital: enc(features_i) → sub_model → enc(emb_i) → W_top_i → enc(logit_i)
      2. Coordinator: enc(logit) = Σ enc(logit_i) + b_top
      3. Patient decrypts enc(logit) and applies exact sigmoid locally
    """

    def __init__(
        self,
        hospital_id: int,
        feature_groups: list,
        emb_dim: int = 16,
        shared_top_W: "nn.Linear | None" = None,
    ):
        super().__init__()
        self.id             = hospital_id
        self.feature_groups = [list(fg) for fg in feature_groups]
        self.emb_dim        = emb_dim
        self.n              = len(feature_groups)
        self.n_features     = sum(len(fg) for fg in feature_groups)
        self.total_emb      = emb_dim * self.n

        self.sub = ClientSubModel(len(feature_groups[hospital_id]), emb_dim)

        if shared_top_W is None:
            self.top_W = nn.Linear(self.total_emb, 1)
        else:
            self.top_W = shared_top_W

        self._he_built = False

    # ── Training helpers ──────────────────────────────────────────────────────

    def local_emb(self, x_local: torch.Tensor) -> torch.Tensor:
        """Sub-model forward on own features. Plaintext, stays local."""
        return self.sub(x_local)

    def logit_share(self, cat_share: torch.Tensor, is_first: bool) -> torch.Tensor:
        """Apply shared top linear (no bias) to a received concat-embedding share.

        sum_j( logit_share_j ) = cat(embs) @ W_top.T + W_top.bias = logit
        Linearity over additive shares removes the need for Beaver Triple.
        """
        logit = F.linear(cat_share, self.top_W.weight)
        if is_first:
            logit = logit + self.top_W.bias
        return logit

    # ── Inference helpers (CKKS) ──────────────────────────────────────────────

    def build_he_weights(
        self,
        W_top_i: "np.ndarray",
        b_top:   "np.ndarray | None" = None,
    ):
        """Pre-compute weight lists for CKKS inference.

        W_top_i : (1, emb_dim) — this hospital's column slice of the top linear:
                  W_top.weight[:, id*emb_dim : (id+1)*emb_dim]
        b_top   : (1,) — shared top-model bias.
                  Every hospital stores it; coordinator adds it once after summing.
        """
        import numpy as np

        # Sub-model first linear — zero-pad input to power-of-2 for TenSEAL mm_
        W_sub  = self.sub.linear.weight.detach().numpy()   # (emb_dim, |fg_i|)
        b_sub  = self.sub.linear.bias.detach().numpy()     # (emb_dim,)
        n_feat = W_sub.shape[1]
        pad_to = max(1 << (n_feat - 1).bit_length(), 4)
        W_pad  = np.zeros((W_sub.shape[0], pad_to), dtype=np.float64)
        W_pad[:, :n_feat] = W_sub
        self._W_sub_T = W_pad.T.astype(np.float64).tolist()   # (pad_to, emb_dim)
        self._b_sub   = b_sub.astype(np.float64).tolist()

        # Sub-model second linear (embed)
        W_embed = self.sub.embed.weight.detach().numpy()   # (emb_dim, emb_dim)
        b_embed = self.sub.embed.bias.detach().numpy()     # (emb_dim,)
        self._W_embed_T = W_embed.T.astype(np.float64).tolist()  # (emb_dim, emb_dim)
        self._b_embed   = b_embed.astype(np.float64).tolist()

        # Top-model column slice for this hospital
        W_arr        = np.array(W_top_i, dtype=np.float64)  # (1, emb_dim)
        self._W_top_T = W_arr.T.astype(np.float64).tolist() # (emb_dim, 1)
        self._b_top   = b_top.astype(np.float64).tolist() if b_top is not None else None

        self._he_built = True

    def compute_sub_emb_he(self, enc_xi_bytes: bytes, ctx) -> bytes:
        """CKKS sub-model: enc(features_i) → enc(embedding_i).

        Linear → CubicAct → Linear  (-2 HE levels total)

        CubicAct: 0.197*z + 0.004*z³ = z * (0.197 + 0.004*z²)
          step 1: z² = z*z          (-1 level)
          step 2: z*(0.197+0.004*z²) (-1 level; TenSEAL auto-modswitch z to z²'s level)
        """
        import tenseal as ts
        assert self._he_built, "call build_he_weights() after loading model"

        enc_z = ts.lazy_ckks_vector_from(enc_xi_bytes)
        enc_z.link_context(ctx)
        enc_z.mm_(self._W_sub_T)
        enc_z += self._b_sub

        # Cubic activation: z * (0.197 + 0.004 * z²)
        enc_z2  = enc_z * enc_z                       # -1 HE level
        enc_act = enc_z * (enc_z2 * 0.004 + 0.197)   # -1 HE level

        enc_act.mm_(self._W_embed_T)
        enc_act += self._b_embed
        return enc_act.serialize()

    def compute_logit_share_he(self, enc_emb_bytes: bytes, ctx) -> bytes:
        """Apply this hospital's W_top column slice to enc(embedding_i) → enc(logit_i).

        No bias added here — coordinator adds b_top once after summing all enc(logit_i).
        """
        import tenseal as ts
        assert self._he_built

        enc_logit = ts.lazy_ckks_vector_from(enc_emb_bytes)
        enc_logit.link_context(ctx)
        enc_logit.mm_(self._W_top_T)
        return enc_logit.serialize()

    def run_he_inference(
        self,
        all_enc_xi: dict,           # {hospital_id: enc_xi_bytes}
        ctx,
        other_hospitals: list,      # list[HospitalModel]
    ) -> bytes:
        """Fully distributed CKKS inference (simulation — all hospitals in one process).

        Protocol:
          1. Each hospital: enc(features_i) → sub_model → enc(emb_i) → W_top_i → enc(logit_i)
          2. Coordinator (self) sums all enc(logit_i) + b_top
          3. Return enc(logit) — patient decrypts and applies sigmoid
        """
        import tenseal as ts
        assert self._he_built

        all_hospitals = sorted([self] + other_hospitals, key=lambda h: h.id)

        enc_logit = None
        for h in all_hospitals:
            enc_emb = h.compute_sub_emb_he(all_enc_xi[h.id], ctx)
            enc_l   = ts.lazy_ckks_vector_from(h.compute_logit_share_he(enc_emb, ctx))
            enc_l.link_context(ctx)
            enc_logit = enc_l if enc_logit is None else enc_logit + enc_l

        if self._b_top is not None:
            enc_logit += self._b_top

        return enc_logit.serialize()

    @staticmethod
    def encrypt_feature_slice(x_slice, ctx) -> bytes:
        """Patient encrypts one hospital's feature slice before sending."""
        import tenseal as ts
        import numpy as np
        return ts.ckks_vector(ctx, x_slice.astype(np.float64).tolist()).serialize()

    @staticmethod
    def decrypt_result(enc_bytes: bytes, ctx) -> float:
        """Patient decrypts enc_logit → sigmoid → probability."""
        import tenseal as ts
        import numpy as np
        vec = ts.lazy_ckks_vector_from(enc_bytes)
        vec.link_context(ctx)
        logit = vec.decrypt()[0]
        return float(1.0 / (1.0 + np.exp(-logit)))
