"""Experimental: does stacking linear-attention blocks (+ LayerNorm to control
the phi(x)=x^2 blow-up) actually improve accuracy on this dataset?

Plain JAX only -- deliberately ignores HE/MPC cost here. The point is to get
an empirical, data-driven answer to "would more depth help" BEFORE paying to
design a Bootstrapped-CKKS inference path and (unsolved, separate) secure
reciprocal-sqrt-under-MPC training path for LayerNorm. If depth doesn't even
help in the unconstrained plaintext setting, it's definitely not worth the
crypto engineering cost; if it does help substantially, that's the signal to
invest in solving the MPC-side LayerNorm problem.

Architecture: same phi(x)=x^2 linear attention as linear_attention_jax.py,
but stacked `num_layers` times with pre-LN + residual connections (standard
"Pre-LN Transformer" pattern, which trains more stably than post-LN).
"""

import jax
import jax.numpy as jnp


def layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def phi(x: jnp.ndarray) -> jnp.ndarray:
    return x ** 2


def sigmoid_approx(logit: jnp.ndarray) -> jnp.ndarray:
    return 0.5 + 0.197 * logit - 0.004 * logit ** 3


def init_params_deep(key, d: int, num_layers: int) -> dict:
    keys = jax.random.split(key, 2 + num_layers * 3)
    scale = 1.0 / jnp.sqrt(d)
    layers = []
    for i in range(num_layers):
        kq, kk, kv = keys[2 + i * 3 : 2 + i * 3 + 3]
        layers.append({
            "W_Q": jax.random.normal(kq, (d, d)) * scale,
            "W_K": jax.random.normal(kk, (d, d)) * scale,
            "W_V": jax.random.normal(kv, (d, d)) * scale,
            "ln_gamma": jnp.ones((d,)),
            "ln_beta":  jnp.zeros((d,)),
        })
    return {
        "cls_token": jax.random.normal(keys[0], (d,)) * scale,
        "layers": layers,
        "final_ln_gamma": jnp.ones((d,)),
        "final_ln_beta":  jnp.zeros((d,)),
        "W_head": jax.random.normal(keys[1], (d, 1)) * scale,
        "b_head": jnp.zeros((1,)),
    }


def linear_attention_block(tokens: jnp.ndarray, p: dict) -> jnp.ndarray:
    """Pre-LN + linear attention + residual."""
    normed = layer_norm(tokens, p["ln_gamma"], p["ln_beta"])
    Q = normed @ p["W_Q"]
    K = normed @ p["W_K"]
    V = normed @ p["W_V"]
    phi_Q, phi_K = phi(Q), phi(K)
    KV       = jnp.einsum("bnd,bne->bde", phi_K, V)
    attn_out = jnp.einsum("bnd,bde->bne", phi_Q, KV)
    return tokens + attn_out  # residual


def forward_deep(feature_tokens: jnp.ndarray, params: dict) -> jnp.ndarray:
    batch = feature_tokens.shape[0]
    cls   = jnp.broadcast_to(params["cls_token"], (batch, 1, feature_tokens.shape[-1]))
    tokens = jnp.concatenate([cls, feature_tokens], axis=1)

    for layer_p in params["layers"]:
        tokens = linear_attention_block(tokens, layer_p)

    # Final LN before the head -- the un-normalized residual stream can grow
    # across layers even though each block's Q/K/V input was pre-LN'd; this
    # was the likely main cause of the earlier catastrophic blow-up (loss in
    # the hundreds of millions) since cls_out fed straight into the cubic
    # sigmoid_approx, which is only valid for |logit| ~< 4.
    tokens  = layer_norm(tokens, params["final_ln_gamma"], params["final_ln_beta"])
    cls_out = tokens[:, 0, :]
    logit   = cls_out @ params["W_head"] + params["b_head"]
    return logit


def loss_deep(feature_tokens: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
    logit = forward_deep(feature_tokens, params)
    pred  = sigmoid_approx(logit)
    return jnp.mean((pred - y) ** 2)


loss_and_grad_deep = jax.value_and_grad(loss_deep, argnums=(0, 2))
