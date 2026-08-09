"""Linear-attention top-model, per linear_attention.md section 3.

Attention(Q,K,V) = phi(Q) @ (phi(K)^T @ V),  phi(x) = x^2   (softmax removed)

Why this shape: computing (phi(K)^T @ V) first -- a (d,d) matrix, contracting
over the N=9 token axis -- then phi(Q) @ that, costs O(N*d^2) instead of the
O(N^2*d) of materializing the full (N,N) attention matrix a softmax version
would need. It also bounds ciphertext x ciphertext multiplications to 3
regardless of N: Q/K/V = tokens @ W_* are ciphertext x PLAINTEXT (free under
CKKS, no level consumed since W_Q/W_K/W_V are known weights, not secret
data); Q^2 and K^2 are 1 level each; (phi(K)^T @ V) and phi(Q) @ (...) are
each ciphertext x ciphertext, 2 more levels -- 3 total, matching
linear_attention.md's "곱셈 Depth 2~3" budget without needing Bootstrapping.

No LayerNorm, no softmax, no FFN block -- every one of those needs either
division/reciprocal-sqrt or an extra nonlinear (ciphertext x ciphertext) op,
which is exactly the CKKS-level/MPC-round budget this design avoids (see
CRYPTEN_MESH_DESIGN.md and the earlier FT-Transformer HE-compatibility
discussion for why softmax/layernorm are the expensive parts).

Sequence layout: token 0 is [CLS] (a learned parameter, not tied to any
single hospital's private data), tokens 1..N are the per-feature tokens from
FeatureTokenizer (model.py), concatenated across hospitals in a fixed order.
"""

import jax
import jax.numpy as jnp


def init_params(key, d: int, n_hospitals_bias_first: bool = True):
    """d = token dim (== emb_dim elsewhere in this project, default 16)."""
    k_cls, k_q, k_k, k_v, k_head = jax.random.split(key, 5)
    scale = 1.0 / jnp.sqrt(d)
    return {
        "cls_token": jax.random.normal(k_cls, (d,)) * scale,
        "W_Q": jax.random.normal(k_q, (d, d)) * scale,
        "W_K": jax.random.normal(k_k, (d, d)) * scale,
        "W_V": jax.random.normal(k_v, (d, d)) * scale,
        "W_head": jax.random.normal(k_head, (d, 1)) * scale,
        "b_head": jnp.zeros((1,)),
    }


def phi(x: jnp.ndarray) -> jnp.ndarray:
    """Linear-attention feature map replacing softmax's exp: phi(x) = x^2."""
    return x ** 2


def sigmoid_approx(logit: jnp.ndarray) -> jnp.ndarray:
    """Same cubic polynomial sigmoid approximation as top_layer_jax.py."""
    return 0.5 + 0.197 * logit - 0.004 * logit ** 3


def linear_attention_forward(feature_tokens: jnp.ndarray, params: dict) -> jnp.ndarray:
    """feature_tokens: (batch, 8, d) -- the 8 per-feature tokens from the
    3 hospitals' FeatureTokenizers, concatenated in a fixed hospital order.
    Returns logit: (batch, 1).
    """
    batch = feature_tokens.shape[0]
    cls   = jnp.broadcast_to(params["cls_token"], (batch, 1, feature_tokens.shape[-1]))
    tokens = jnp.concatenate([cls, feature_tokens], axis=1)  # (batch, 9, d)

    Q = tokens @ params["W_Q"]   # (batch, 9, d) -- cipher x plaintext, free HE level
    K = tokens @ params["W_K"]
    V = tokens @ params["W_V"]

    phi_Q, phi_K = phi(Q), phi(K)                              # 1 HE level each
    KV       = jnp.einsum("bnd,bne->bde", phi_K, V)            # 1 more level
    attn_out = jnp.einsum("bnd,bde->bne", phi_Q, KV)           # 1 more level (3 total)

    cls_out = attn_out[:, 0, :]                                # (batch, d)
    logit   = cls_out @ params["W_head"] + params["b_head"]    # cipher x plaintext, free
    return logit


def linear_attention_loss(feature_tokens: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
    """MSE loss, matching top_layer_jax.py's rationale: BCE's gradient needs
    a secure-division protocol; MSE's is a plain subtraction."""
    logit = linear_attention_forward(feature_tokens, params)
    pred  = sigmoid_approx(logit)
    diff  = pred - y
    return jnp.mean(diff ** 2)


linear_attention_loss_and_grad = jax.value_and_grad(linear_attention_loss, argnums=(0, 2))
"""Returns (loss, (d(loss)/d(feature_tokens), d(loss)/d(params))).

d(loss)/d(feature_tokens) is what would get party-scoped-revealed back to
each hospital (sliced to that hospital's own token range) to continue
backward into its local FeatureTokenizer -- same pattern as
top_layer_jax.py's dL/d(logit_share_i), just token-shaped instead of scalar.
"""


if __name__ == "__main__":
    import numpy as np

    key = jax.random.PRNGKey(0)
    d, n_features, batch = 16, 8, 8

    params = init_params(key, d)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (batch, n_features, d))
    y      = (np.random.default_rng(0).random((batch, 1)) > 0.5).astype(np.float32)

    loss, (grad_tokens, grad_params) = linear_attention_loss_and_grad(tokens, jnp.array(y), params)

    print("=== Linear attention top-model smoke test ===")
    print(f"  tokens shape: {tokens.shape}  ->  logit shape: {linear_attention_forward(tokens, params).shape}")
    print(f"  loss: {float(loss):.6f}")
    print(f"  grad_tokens shape: {grad_tokens.shape} (should match tokens shape {tokens.shape})")
    for name, g in grad_params.items():
        print(f"  grad[{name}] shape: {g.shape}, norm: {float(jnp.linalg.norm(g)):.4f}")

    # Finite-difference check on one param to catch a wrong einsum/shape bug.
    eps = 1e-4
    probe = ("W_head", (0, 0))
    p_plus  = {k: (v.at[probe[1]].add(eps) if k == probe[0] else v) for k, v in params.items()}
    p_minus = {k: (v.at[probe[1]].add(-eps) if k == probe[0] else v) for k, v in params.items()}
    loss_plus  = linear_attention_loss(tokens, jnp.array(y), p_plus)
    loss_minus = linear_attention_loss(tokens, jnp.array(y), p_minus)
    numeric_grad = float((loss_plus - loss_minus) / (2 * eps))
    analytic_grad = float(grad_params[probe[0]][probe[1]])
    print(f"\n  finite-diff check on {probe}: numeric={numeric_grad:.6f}  analytic={analytic_grad:.6f}"
          f"  (should match closely)")
