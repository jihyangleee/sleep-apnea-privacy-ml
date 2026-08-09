"""Top-layer (logit combine -> sigmoid approx -> loss) expressed in plain JAX.

Why this file exists
---------------------
`secret_sharing.py`'s `BeaverProvider` hand-rolls the MPC protocol for every
secret x secret multiplication (Beaver Triple: mask, broadcast, reconstruct).
That's necessary when you implement MPC yourself, but it's exactly the kind
of thing SecretFlow's SPU device does for you automatically: when this same
JAX code is later traced and compiled for the SPU device, every `*` between
two SPU-resident (secret-shared) values is transparently replaced by a secure
multiplication protocol by the SPU compiler. You do not write Beaver Triple
calls by hand for SPU -- you write the plaintext-looking formula once, and
the security comes from *where* (which device) the inputs live when the
function runs, not from how the formula is written.

This module is the plaintext-math prototype: same formulas as
`secret_sharing.py`'s `BeaverProvider.sigmoid_approx` + the MSE loss in
`simulate.py`, rewritten as ordinary JAX functions with no MPC in them yet.
It runs today on plain JAX (verified to install and run natively on Windows).
Wiring it onto `sf.spu` (SecretFlow's SPU device) is a follow-up step that
requires WSL2, since SecretFlow itself doesn't support native Windows.

Scope: only the part that actually needs multi-hospital secrecy protection --
combining logit shares, the sigmoid polynomial approximation, and the MSE
loss against secret-shared labels. `logit_share_i = W_top_i @ emb_i` and the
sub-model stay local, plaintext PyTorch (block-linearity means no other
hospital's data is involved in that matmul) -- see CRYPTEN_MESH_DESIGN.md
section 3 for why the boundary is drawn here.
"""

import jax
import jax.numpy as jnp


def compute_logit_share(local_embedding: jnp.ndarray, W_slice: jnp.ndarray,
                         bias: "jnp.ndarray | None" = None) -> jnp.ndarray:
    """logit_share_i = local_embedding @ W_slice.T (+ bias for exactly one hospital).

    Included for completeness / to match CRYPTEN_MESH_DESIGN.md's boundary
    table (this op is listed as the SPU/CrypTen entry point). In practice it
    never needs protection on its own -- block-linearity means only hospital
    i's own data is involved -- so it's fine to keep this in local PyTorch
    and only feed the resulting scalar `logit_share_i` into the functions
    below. Kept here as a JAX equivalent in case the whole top-layer is
    later traced as a single SPU-compiled function.
    """
    logit = local_embedding @ W_slice.T
    if bias is not None:
        logit = logit + bias
    return logit


def sigmoid_approx(logit: jnp.ndarray) -> jnp.ndarray:
    """sigma(x) ~= 0.5 + 0.197x - 0.004x^3.

    Identical formula to `secret_sharing.py`'s `BeaverProvider.sigmoid_approx`.
    There, the same polynomial is evaluated share-by-share through two Beaver
    Triple multiplications so that no party ever reconstructs `logit`. Here
    it's evaluated directly on the (already-summed) plaintext logit -- valid
    today because we're only proving out the math, not running it under MPC
    yet. When the SAME function is later called with `logit` living on the
    SPU device (secret-shared across 3 hospitals), the `logit * logit` and
    `(...) * logit` multiplications below become secure multiplications
    automatically, with no code change here.
    """
    return 0.5 + 0.197 * logit - 0.004 * logit ** 3


def top_layer_forward(logit_shares: list) -> jnp.ndarray:
    """sum_j(logit_share_j) = logit  (additive shares from block-linearity,
    see HospitalModel.logit_share in model.py), then sigmoid approx.
    """
    logit = sum(logit_shares)
    return sigmoid_approx(logit)


def top_layer_loss(logit_shares: list, y_shares: list) -> jnp.ndarray:
    """MSE loss against secret-shared labels -- mirrors simulate.py Step 3-4.

    diff = pred - y = sigmoid_approx(sum(logit_shares)) - sum(y_shares)
    loss = mean(diff^2)
    """
    pred = top_layer_forward(logit_shares)
    y    = sum(y_shares)
    diff = pred - y
    return jnp.mean(diff ** 2)


# Differentiable end to end via jax.grad/jax.value_and_grad -- this is the
# JAX-side analogue of CrypTen's AutogradCrypTensor: once this function runs
# under SPU, calling jax.grad on it yields dL/d(logit_share_i) computed
# entirely inside the secure domain, revealed only via SPU's party-scoped
# reveal to hospital i (see CRYPTEN_MESH_DESIGN.md section 4).
top_layer_loss_and_grad = jax.value_and_grad(
    lambda logit_shares, y_shares: top_layer_loss(logit_shares, y_shares),
    argnums=0,
)


if __name__ == "__main__":
    import numpy as np
    import torch
    from secret_sharing import additive_split, BeaverProvider

    rng = np.random.default_rng(0)
    batch, n_parties = 8, 3

    local_logit_shares_np = [rng.normal(size=(batch, 1)).astype(np.float32) for _ in range(n_parties)]
    y_np = (rng.random(size=(batch, 1)) > 0.5).astype(np.float32)

    # ── Reference: existing PyTorch + BeaverProvider path (secret_sharing.py) ──
    logit_shares_t = [torch.tensor(s, requires_grad=True) for s in local_logit_shares_np]
    y_t            = torch.tensor(y_np)
    beaver         = BeaverProvider(n_parties)
    triple1        = beaver.generate_triple(logit_shares_t[0].shape)
    triple2        = beaver.generate_triple(logit_shares_t[0].shape)
    pred_shares_t  = beaver.sigmoid_approx(logit_shares_t, triple1, triple2)
    y_shares_t     = additive_split(y_t, n=n_parties)
    diff_shares_t  = [pred_shares_t[i] - y_shares_t[i] for i in range(n_parties)]
    diff_t         = sum(diff_shares_t)
    loss_t         = (diff_t ** 2).mean()
    loss_t.backward()
    grads_t = [g.grad.detach().numpy() for g in logit_shares_t]

    # ── New: plain JAX path (top_layer_jax.py) ─────────────────────────────────
    # Only sum(y_shares) matters for the result, so an even split is a valid
    # (if non-random) secret sharing of y for this parity check.
    logit_shares_j = [jnp.array(s) for s in local_logit_shares_np]
    y_shares_j     = [jnp.array(y_np) / n_parties for _ in range(n_parties)]

    loss_j, grads_j = top_layer_loss_and_grad(logit_shares_j, y_shares_j)

    print("=== Forward parity (loss) ===")
    print(f"  PyTorch+Beaver loss : {loss_t.item():.6f}")
    print(f"  Plain JAX loss      : {float(loss_j):.6f}")

    print("=== Backward parity (dL/d(logit_share_i)) ===")
    for i in range(n_parties):
        print(f"  hospital {i}: torch={grads_t[i].ravel()[:3]}  jax={np.array(grads_j[i]).ravel()[:3]}")
