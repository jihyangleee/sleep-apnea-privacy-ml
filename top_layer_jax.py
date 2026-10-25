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


# ── BCE-safe variant (sigma'(z) floor) ──────────────────────────────────────
# 평문 실험(experiments/features/mse_vs_bce.py의 bce_sigprime_floor)에서 가장
# 좋은 성능을 낸 그래디언트를 SPU 경로에 그대로 옮긴 것.
#
#   grad_z = (pred - y) * max(sigma'(z), floor) / batch
#
# MSE의 자동미분 그래디언트는 2*(pred-y)*sigma'(z)/batch 인데, sigma'(z) =
# 0.197 - 0.012*z^2 가 |z|~4.05를 넘으면 음수로 뒤집혀 그래디언트 방향이
# 틀어진다 (지난 분석 참고). 여기서는 sigma'(z)에 바닥(floor)을 둬서 방향이
# 절대 뒤집히지 않게 한다. jax.grad로 이 식을 "유도"하는 게 아니라
# jax.custom_vjp로 역전파를 직접 지정한다 -- autodiff가 자동으로 만들어내는
# 진짜 MSE 그래디언트가 아니라, 의도적으로 다른 그래디언트를 주입하는 것이기
# 때문이다 (지난 설명의 "근사의 근사"에 해당).
#
# SPU에서 돌 때는 jnp.clip이 비교(comparison) 연산으로 컴파일된다 -- ABY3가
# 비교를 기본 프리미티브로 지원하므로 추가 구현 없이 그대로 동작한다.
def _bce_safe_core(logit: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = sigmoid_approx(logit)
    return jnp.mean((pred - y) ** 2)


_bce_safe_core = jax.custom_vjp(_bce_safe_core)


def _bce_safe_core_fwd(logit, y):
    pred = sigmoid_approx(logit)
    loss = jnp.mean((pred - y) ** 2)  # 모니터링용 -- 역전파에는 쓰이지 않음
    return loss, (logit, y, pred)


def _bce_safe_core_bwd(res, g):
    logit, y, pred = res
    sigma_prime = 0.197 - 0.012 * logit ** 2
    weight = jnp.clip(sigma_prime, a_min=0.02)  # 바닥 -- 여기서 비교(clip) 발생
    batch = logit.shape[0]
    grad_logit = g * (pred - y) * weight / batch
    return (grad_logit, jnp.zeros_like(y))


_bce_safe_core.defvjp(_bce_safe_core_fwd, _bce_safe_core_bwd)


def top_layer_loss_bce_safe(logit_shares: list, y_shares: list) -> jnp.ndarray:
    logit = sum(logit_shares)
    y = sum(y_shares)
    return _bce_safe_core(logit, y)


top_layer_loss_and_grad_bce_safe = jax.value_and_grad(
    lambda logit_shares, y_shares: top_layer_loss_bce_safe(logit_shares, y_shares),
    argnums=0,
)


# ── MSE-floor variant (실측 1등: MSE 계수 2 유지 + sigma'(z) 바닥) ──────────────
# experiments/features/bce_safe_15features_selfsim.py의 mse_floor와 동일 공식.
# grad_z = 2*(pred-y)*max(sigma'(z), floor)/batch
def _mse_floor_core(logit: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = sigmoid_approx(logit)
    return jnp.mean((pred - y) ** 2)


_mse_floor_core = jax.custom_vjp(_mse_floor_core)


def _mse_floor_core_fwd(logit, y):
    pred = sigmoid_approx(logit)
    loss = jnp.mean((pred - y) ** 2)
    return loss, (logit, y, pred)


def _mse_floor_core_bwd(res, g):
    logit, y, pred = res
    sigma_prime = 0.197 - 0.012 * logit ** 2
    weight = jnp.clip(sigma_prime, a_min=0.02)
    batch = logit.shape[0]
    grad_logit = g * 2 * (pred - y) * weight / batch
    return (grad_logit, jnp.zeros_like(y))


_mse_floor_core.defvjp(_mse_floor_core_fwd, _mse_floor_core_bwd)


def top_layer_loss_mse_floor(logit_shares: list, y_shares: list) -> jnp.ndarray:
    logit = sum(logit_shares)
    y = sum(y_shares)
    return _mse_floor_core(logit, y)


top_layer_loss_and_grad_mse_floor = jax.value_and_grad(
    lambda logit_shares, y_shares: top_layer_loss_mse_floor(logit_shares, y_shares),
    argnums=0,
)


# ── DP 노이즈를 reveal 이전(SPU 내부)에서 더하는 버전 ───────────────────────────
# 문제: 기존 Algorithm 1은 g를 병원 i에게 공개한 "뒤" 노이즈를 더한다. 이 순서에서는
# 병원 i가 노이즈 없는 진짜 g를 먼저 보게 되므로, g의 부호로 라벨을 그대로 읽을 수
# 있다 -- 노이즈가 병원 i 자신으로부터는 전혀 보호하지 못한다.
#
# 해법: 노이즈 자체를 비밀분산해서 SPU 안으로 들여보낸 뒤, 그래디언트에 더하고 나서
# reveal한다. 이러면 공개되는 순간 이미 노이즈가 섞여 있어서, 그 노이즈의 합을
# 혼자 아는 병원이 없으므로 (라벨 y를 비밀분산하는 것과 완전히 같은 패턴) 정확한
# 원본 g를 아무도 복원할 수 없다.
#
# noise_shares: 3개 병원이 각자 로컬에서 N(0, sigma_dp^2/3)을 뽑아 만든 조각.
# 세 조각의 합은 N(0, sigma_dp^2) 전체이며, 이 합을 혼자 아는 병원은 없다
# (병원 i는 자기 조각만 알고, 남의 조각은 share로만 받는다 -- additive_split과 동일 패턴).
# 노이즈가 섞인 g+noise는 전 병원에게 동일하게 공개된다 (병원별로 다른 노이즈를
# 따로 공개하려면 "병원 i를 제외한 나머지 둘이" 그 노이즈를 대신 넣어줘야 하는데,
# 이 버전은 가장 단순한 형태로 전원에게 같은 노이즈 낀 값을 공개하는 쪽을 택했다).
def _loss_and_noisy_grad_mse_floor(logit_shares: list, y_shares: list, noise_shares: list):
    loss, grads = top_layer_loss_and_grad_mse_floor(logit_shares, y_shares)
    noise_total = sum(noise_shares)  # SPU 내부에서만 합쳐짐 -- 어느 병원도 이 합을 혼자 모름
    noisy_grads = [g + noise_total for g in grads]
    return (loss, *noisy_grads)


# ── BCE 로짓 트릭 + z 범위 제한 (clip_z / seg3) ─────────────────────────────
# 평문 실험(mse_vs_bce.py)의 bce_clip_z / bce_seg3 를 SPU 경로로 옮긴 것.
#
#   grad_z = (sigma_hat(z) - y) / batch
#
#   clip_z : sigma_hat(z) = sigmoid_approx(clip(z, -4, 4))
#            3차 근사의 기울기 0.197-0.012z^2 가 |z|~4.05에서 0이 되므로
#            그 안쪽(단조증가 구간)으로 z를 먼저 묶는다.
#   seg3   : sigma_hat(z) = SPU SIGMOID_SEG3 (|z|<=4: 0.5+0.125z, 바깥: 0/1)
#
# bce_safe와 마찬가지로 jax.custom_vjp로 역전파를 직접 지정하고, forward의 loss는
# 모니터링용 MSE 값이다. clip/select는 SPU에서 비교 연산으로 컴파일된다.
def _sigmoid_clip_z(logit):
    return sigmoid_approx(jnp.clip(logit, -4.0, 4.0))
 # 0.5 + 0.197z − 0.004z³ 로 sigmoid를 근사함 -> 약 범위가 0.06 에서 0.94가 나옴
def _sigmoid_seg3(logit):
    return jnp.select([logit < -4, logit > 4], [0.0, 1.0], 0.5 + 0.125 * logit)


def _make_bce_logit_loss(sigma_hat):
    @jax.custom_vjp
    def core(logit, y): # loss 식 
        return jnp.mean((sigma_hat(logit) - y) ** 2)

    def fwd(logit, y):
        pred = sigma_hat(logit)
        return jnp.mean((pred - y) ** 2), (y, pred) 

    def bwd(res, g): # 역전파 - gradient를 유도하지 않고 직접 지정함 
        y, pred = res
        grad_logit = g * (pred - y) / pred.shape[0] 
        return (grad_logit, jnp.zeros_like(y)) 

    core.defvjp(fwd, bwd)  # 미분은 bwd 로 하는 뜻 
    return jax.value_and_grad(
        lambda logit_shares, y_shares: core(sum(logit_shares), sum(y_shares)),
        argnums=0,
    )


top_layer_loss_and_grad_bce_clip_z = _make_bce_logit_loss(_sigmoid_clip_z) 
# 여기서 loss 와 gradient를 구함
top_layer_loss_and_grad_bce_seg3 = _make_bce_logit_loss(_sigmoid_seg3)


# bce_clip_z에 _loss_and_noisy_grad_mse_floor와 같은 "reveal 이전 노이즈"를 적용한 버전.
# grad_z = (sigma_hat(z) - y) / batch 도 부호로 라벨이 드러나므로 같은 방식으로 막는다.
def _loss_and_noisy_grad_bce_clip_z(logit_shares: list, y_shares: list, noise_shares: list):
    loss, grads = top_layer_loss_and_grad_bce_clip_z(logit_shares, y_shares) 
    # loss 와 gradients 3개를 구한다.(z_A,z_B,z_C)
    noise_total = sum(noise_shares)  # SPU 내부에서만 합쳐짐 -- 어느 병원도 이 합을 혼자 모름
    noisy_grads = [g + noise_total for g in grads] # 같은 noise(dp)를 gradient에 더한다. 
    return (loss, *noisy_grads)


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
