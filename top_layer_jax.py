"""SPU 학습의 top layer: logit 조각 합 → 근사 시그모이드 → BCE 그래디언트 + DP 노이즈.

mpc/train_spu_party.py가 이 함수를 SPU 디바이스 위에서 실행한다. 입력이 SPU에 비밀분산돼
있으면 아래의 곱셈·비교는 SPU 컴파일러가 MPC 프로토콜(ABY3)로 바꿔 실행하므로, 식은
평문 JAX로 한 번만 쓰면 된다.

  z      = Σ logit_share_i
  p̂      = σ̂(clip(z, −4, 4)),   σ̂(z) = 0.5 + 0.197z − 0.004z³
  grad_z = (p̂ − y) / batch + Σ noise_i      ← 노이즈를 더한 뒤에야 공개된다
"""

import jax
import jax.numpy as jnp


def sigmoid_approx(logit: jnp.ndarray) -> jnp.ndarray:
    """σ(z) ≈ 0.5 + 0.197z − 0.004z³. MPC는 지수 함수를 싸게 계산하지 못해 3차 다항식으로 근사한다.
    |z| ≤ 4에서 실제 시그모이드와 가깝고, 출력 범위는 약 0.06~0.94다."""
    return 0.5 + 0.197 * logit - 0.004 * logit ** 3


def _sigmoid_clip_z(logit):
    # σ̂의 기울기 0.197 − 0.012z²는 |z| ≈ 4.05 밖에서 음수가 되므로, 단조증가 구간 안으로 z를 먼저 자른다.
    # SPU에서 jnp.clip은 비교 연산으로 컴파일된다.
    return sigmoid_approx(jnp.clip(logit, -4.0, 4.0))


def _make_bce_logit_loss(sigma_hat):
    """BCE의 로짓 그래디언트 σ(z) − y에서 σ만 σ̂로 바꾼 손실.

    jax.custom_vjp로 역전파를 직접 지정한다 (자동미분이 만드는 MSE 그래디언트가 아니다).
    forward가 돌려주는 loss는 모니터링용 MSE 값이다.
    """
    @jax.custom_vjp
    def core(logit, y):
        return jnp.mean((sigma_hat(logit) - y) ** 2)

    def fwd(logit, y):
        pred = sigma_hat(logit)
        return jnp.mean((pred - y) ** 2), (y, pred)

    def bwd(res, g):
        y, pred = res
        return (g * (pred - y) / pred.shape[0], jnp.zeros_like(y))

    core.defvjp(fwd, bwd)
    return jax.value_and_grad(
        lambda logit_shares, y_shares: core(sum(logit_shares), sum(y_shares)),
        argnums=0,
    )


top_layer_loss_and_grad_bce_clip_z = _make_bce_logit_loss(_sigmoid_clip_z)


def _loss_and_noisy_grad_bce_clip_z(logit_shares: list, y_shares: list, noise_shares: list):
    """loss와 병원별 grad_z를 계산하고, 공개 전에 DP 노이즈를 더한다.

    grad_z = (σ̂(z) − y) / batch는 부호로 라벨이 드러난다. 그래서 세 병원이 각자 뽑은
    N(0, σ²/3) 조각(noise_shares)을 SPU 안에서 합쳐 그래디언트에 더한 뒤에 공개한다.
    노이즈 합을 혼자 아는 병원은 없다.
    """
    loss, grads = top_layer_loss_and_grad_bce_clip_z(logit_shares, y_shares)
    noise_total = sum(noise_shares)
    return (loss, *[g + noise_total for g in grads])
