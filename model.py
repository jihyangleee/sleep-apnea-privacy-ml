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


class FeatureTokenizer(nn.Module):
    """FT-Transformer style per-feature tokenizer (linear_attention.md §3.1).

    Unlike ClientSubModel, which collapses a hospital's features into a
    single emb_dim embedding, this keeps one token PER feature so the
    top-model can attend across all 8 features (+ [CLS]) as a 9-token
    sequence. Standard numerical-feature tokenization (Gorishniy et al.,
    "Revisiting Deep Learning Models for Tabular Data", 2021):
        token_i = bias_i + x_i * weight_i
    i.e. each feature gets its own learned (weight, bias) pair projecting
    its scalar value into a d-dim token — a per-feature nn.Linear(1, d)
    implemented as a single batched weight/bias for all of this hospital's
    features at once.

    Stays local, plaintext PyTorch — like ClientSubModel, this never needs
    MPC/HE protection since it only touches this hospital's own features.
    """
    def __init__(self, n_features: int, d: int = 16):
        super().__init__()
        self.n_features = n_features
        self.d          = d
        self.weight = nn.Parameter(torch.randn(n_features, d) / d ** 0.5)
        self.bias   = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features) -> tokens: (batch, n_features, d)"""
        return x.unsqueeze(-1) * self.weight + self.bias


class HospitalModel(nn.Module):
    """Per-hospital model: private sub-model + own column slice of top-model.

    Training  — each hospital applies its own W_top column-slice to its own
                local embedding; the embedding never leaves the hospital.
                The resulting per-hospital logits are already additive
                shares of the true logit (block-linearity of W_top over the
                concatenation), so Beaver Triple is only needed to secure
                the nonlinear sigmoid computed over those shares.
    Inference — CKKS ciphertext arithmetic (patient features never decrypted)

    Both protocols follow the same shape (3 hospitals, 1 round):
      1. Each hospital: features_i → sub_model → emb_i → W_top_i → logit_i
      2. Coordinator: logit = Σ logit_i + b_top  (plaintext sum for training,
         ciphertext sum for inference)
      3. Sigmoid: Beaver Triple approximation (training) or exact after
         decrypt (inference)
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

    def logit_share(self, local_embedding: torch.Tensor, is_first: bool) -> torch.Tensor:
        """Apply this hospital's own column-slice of the shared top linear to
        its own local embedding — no cross-hospital embedding exchange needed.

        W_top @ concat(embs) decomposes over the block structure of the
        concatenation:
            sum_j( W_top[:, j*emb_dim:(j+1)*emb_dim] @ emb_j ) + bias = logit
        so each hospital only ever needs its own embedding and its own
        weight columns. The resulting logit_share is already an additive
        share of the true logit — no Beaver Triple needed for this step.
        """
        start   = self.id * self.emb_dim
        W_slice = self.top_W.weight[:, start : start + self.emb_dim]
        logit   = F.linear(local_embedding, W_slice)
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
        import tenseal as ts #동형암호 라이브러리 
        assert self._he_built, "call build_he_weights() after loading model"

        enc_z = ts.lazy_ckks_vector_from(enc_xi_bytes) # bytes를 객체로 만들어줌 
        enc_z.link_context(ctx) # ctx 정보를 활용 
        enc_z.mm_(self._W_sub_T)  # submodel 에서 연산 
        enc_z += self._b_sub # 절편값 더함

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

        enc_logit = ts.lazy_ckks_vector_from(enc_emb_bytes) #embedding 값을 부여받음 
        enc_logit.link_context(ctx)
        enc_logit.mm_(self._W_top_T) # top model 과 연산 
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
