"""14개 피처(변형 조합: times34p 포함, sleep_latency 제외)로 SPU 경로를 학습한다.

simulate_spu.py의 run_distributed_simulation_spu와 같은 SPU 프로토콜(ABY3)을 쓰되,
15개 고정 SLEEP_FEATURE_GROUPS 대신 14개 피처 + 역할 기반 분배를 쓴다.
train_variant_features.py(자체 Beaver 시뮬레이션 버전)와 같은 피처 구성이라 결과를
직접 비교할 수 있다.

WSL 전용 (venv_wsl, secretflow).
사용법: python -m mpc.train_variant_spu --epochs 30 --max-batches 1   (스모크 테스트)
        python -m mpc.train_variant_spu --epochs 100                 (본 실행)
"""

import argparse
import time

import numpy as np
import torch
import jax.numpy as jnp
import secretflow as sf
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from sklearn.model_selection import train_test_split

from dataset import load_shhs_data, MODEL_FEATURES
from model import LinearHospitalModel
from secret_sharing import additive_split
from top_layer_jax import (top_layer_loss_and_grad, top_layer_loss_and_grad_bce_safe,
                           top_layer_loss_and_grad_bce_clip_z, top_layer_loss_and_grad_bce_seg3,
                           top_layer_loss_and_grad_mse_floor, _loss_and_noisy_grad_mse_floor)
from mpc.simulate_spu import _make_spu


def _loss_and_per_party_grads(logit_shares, y_shares):
    loss, grads = top_layer_loss_and_grad(logit_shares, y_shares)
    return (loss, *grads)


def _loss_and_per_party_grads_bce_safe(logit_shares, y_shares):
    loss, grads = top_layer_loss_and_grad_bce_safe(logit_shares, y_shares)
    return (loss, *grads)


def _loss_and_per_party_grads_bce_clip_z(logit_shares, y_shares):
    loss, grads = top_layer_loss_and_grad_bce_clip_z(logit_shares, y_shares)
    return (loss, *grads)


def _loss_and_per_party_grads_bce_seg3(logit_shares, y_shares):
    loss, grads = top_layer_loss_and_grad_bce_seg3(logit_shares, y_shares)
    return (loss, *grads)


def _loss_and_per_party_grads_mse_floor(logit_shares, y_shares):
    loss, grads = top_layer_loss_and_grad_mse_floor(logit_shares, y_shares)
    return (loss, *grads)


# mse_floor_dp 는 2개 인자(logit_shares, y_shares)가 아니라 noise_shares까지 3개를
# 받는다 -- 일반 LOSS_FNS 딕셔너리로 통일하지 않고 run()에서 따로 분기한다.
LOSS_FNS = {
    "mse": _loss_and_per_party_grads,
    "bce_safe": _loss_and_per_party_grads_bce_safe,
    "bce_clip_z": _loss_and_per_party_grads_bce_clip_z,
    "bce_seg3": _loss_and_per_party_grads_bce_seg3,
    "mse_floor": _loss_and_per_party_grads_mse_floor,
}
DP_INSIDE_SPU_LOSS = "mse_floor_dp"
ALL_LOSS_CHOICES = list(LOSS_FNS) + [DP_INSIDE_SPU_LOSS]

NUM_CLIENTS = 3
AHI_THRESHOLD = 15

ROLE_FEATURES = {
    "A": ["avgsat", "avg_hr", "odi", "mean_desat_duration", "mean_desat_drop"],
    "B": ["slpprdp", "slpeffp", "times34p", "waso"],
    "C": ["age_s1", "gender", "bmi_s1", "neck20", "ess_s1"],
}


def build_14_feature_data(csv_path):
    features = [f for role in ROLE_FEATURES.values() for f in role]
    assert len(features) == 14
    col_idx = [MODEL_FEATURES.index(f) for f in features]
    groups, start = [], 0
    for role in ROLE_FEATURES.values():
        groups.append(list(range(start, start + len(role))))
        start += len(role)

    X15, y = load_shhs_data(csv_path, return_scaler=False)
    X = X15[:, col_idx]
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    return X[idx_train], y[idx_train], X[idx_test], y[idx_test], groups, features


def run(csv_path, n_epochs, batch_size, max_batches_per_epoch, dp_sigma, lr, eval_every, seed=0, loss_type="mse",
        reveal_loss=False):
    dp_inside_spu = (loss_type == DP_INSIDE_SPU_LOSS)
    loss_fn = None if dp_inside_spu else LOSS_FNS[loss_type]
    # mse_floor_dp 에서는 loss도 기본적으로 비공개다 (reveal_loss=True일 때만 디버깅용으로 공개).
    # 다른 loss_type은 기존처럼 항상 공개(모니터링용) -- 이쪽은 애초에 그래디언트도
    # 공개-후-노이즈라 "평문 유출 없음"을 주장하지 않는 비교용 베이스라인이다.
    show_loss = reveal_loss if dp_inside_spu else True
    print(f"[SPU] loss_type={loss_type}"
          + ("  (DP 노이즈 reveal 이전, SPU 내부에서 적용)" if dp_inside_spu else "")
          + ("  (loss 비공개)" if dp_inside_spu and not reveal_loss else ""))
    torch.manual_seed(seed)
    X_train, y_train_raw, X_test, y_test_raw, feature_groups, features = build_14_feature_data(csv_path)
    print(f"[data] 14개 피처: {features}")
    print(f"[groups] {feature_groups}")
    print(f"[data] train n={len(X_train)}, test n={len(X_test)}")

    hospitals = [LinearHospitalModel(i, feature_groups) for i in range(NUM_CLIENTS)]
    X_trains = [torch.tensor(X_train[:, g], dtype=torch.float32) for g in feature_groups]
    X_tests  = [torch.tensor(X_test[:, g],  dtype=torch.float32) for g in feature_groups]
    y_train  = torch.tensor(y_train_raw, dtype=torch.float32)
    y_test   = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)

    params = [h.sub.weight for h in hospitals] + [hospitals[0].bias]
    optimizer = torch.optim.Adam(params, lr=lr)

    print("[SPU] secretflow 3-party SPU 초기화 (ABY3, semi-honest, honest-majority)")
    spu, pyus = _make_spu()

    t_train_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0
        t_epoch_start = time.perf_counter()

        for start in range(0, n, batch_size):
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                break
            idx = perm[start:start + batch_size]
            if len(idx) < batch_size:
                break
            y_batch = y_train[idx]
            optimizer.zero_grad()

            local_embs = [hospitals[i].local_emb(X_trains[i][idx]) for i in range(NUM_CLIENTS)]
            logit_shares = [hospitals[j].logit_share(local_embs[j], j == 0) for j in range(NUM_CLIENTS)]
            logit_shares_np = [ls.detach().numpy() for ls in logit_shares]
            y_shares_np = [s.numpy() for s in additive_split(y_batch.unsqueeze(1), n=NUM_CLIENTS)]

            logit_shares_pyu = [pyus[i](lambda x: jnp.array(x))(logit_shares_np[i]) for i in range(NUM_CLIENTS)]
            y_shares_pyu = [pyus[i](lambda x: jnp.array(x))(y_shares_np[i]) for i in range(NUM_CLIENTS)]
            logit_shares_spu = [x.to(spu) for x in logit_shares_pyu]
            y_shares_spu = [x.to(spu) for x in y_shares_pyu]

            if dp_inside_spu:
                # 각 병원이 로컬에서 N(0, dp_sigma^2/3)을 뽑는다. 세 조각의 합은
                # N(0, dp_sigma^2) 전체이고, 그 합을 혼자 아는 병원은 없다
                # (라벨을 additive_split하는 것과 완전히 같은 패턴).
                per_party_sigma = dp_sigma / np.sqrt(NUM_CLIENTS) if dp_sigma > 0 else 0.0
                noise_shares_np = [
                    np.random.normal(0.0, per_party_sigma, size=logit_shares_np[0].shape).astype(np.float32)
                    for _ in range(NUM_CLIENTS)
                ]
                noise_shares_pyu = [pyus[i](lambda x: jnp.array(x))(noise_shares_np[i]) for i in range(NUM_CLIENTS)]
                noise_shares_spu = [x.to(spu) for x in noise_shares_pyu]

                outs = spu(
                    _loss_and_noisy_grad_mse_floor,
                    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                    user_specified_num_returns=1 + NUM_CLIENTS,
                )(logit_shares_spu, y_shares_spu, noise_shares_spu)
                loss_spu, grads_spu = outs[0], list(outs[1:])

                # show_loss=False면 loss_spu는 끝까지 비밀분산 상태로 남고 reveal() 자체를
                # 호출하지 않는다 -- "값이 있는데 숨긴다"가 아니라 "애초에 평문으로 만들지
                # 않는다"는 차이다.
                loss = float(sf.reveal(loss_spu)) if show_loss else float("nan")
                # 이미 노이즈가 섞인 채로 공개되므로, 공개 이후에 또 노이즈를 더하지 않는다.
                grads = sf.reveal([grads_spu[i].to(pyus[i]) for i in range(NUM_CLIENTS)])
                for i in range(NUM_CLIENTS):
                    grad_i = torch.tensor(np.array(grads[i]), dtype=torch.float32)
                    logit_shares[i].backward(grad_i)
            else:
                outs = spu(
                    loss_fn,
                    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                    user_specified_num_returns=1 + NUM_CLIENTS,
                )(logit_shares_spu, y_shares_spu)
                loss_spu, grads_spu = outs[0], list(outs[1:])

                loss = float(sf.reveal(loss_spu))
                grads = sf.reveal([grads_spu[i].to(pyus[i]) for i in range(NUM_CLIENTS)])
                for i in range(NUM_CLIENTS):
                    grad_i = torch.tensor(np.array(grads[i]), dtype=torch.float32)
                    if dp_sigma > 0:
                        # 참고: 이 경로는 여전히 "공개 후" 노이즈 (옛 Algorithm 1과 동일,
                        # mse/bce_safe 등 비교용 베이스라인에만 씀). 실제 배포 기준은
                        # loss_type=mse_floor_dp 를 쓴다.
                        grad_i = grad_i + torch.randn_like(grad_i) * dp_sigma
                    logit_shares[i].backward(grad_i)
            optimizer.step()

            if show_loss:
                epoch_loss += loss
            n_batches += 1

        epoch_wall_s = time.perf_counter() - t_epoch_start
        loss_str = f"{epoch_loss/max(n_batches,1):.4f}" if show_loss else "비공개"
        msg = f"  Epoch {epoch:3d}/{n_epochs} | loss={loss_str} | wall={epoch_wall_s:.1f}s ({n_batches} batches)"

        if epoch % eval_every == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                logit = sum(hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)) + hospitals[0].bias
                pred = (logit.squeeze() > 0).float()
                acc = (pred == y_test).float().mean().item()
            msg += f" | test_acc={acc:.4f}"
        print(msg, flush=True)

    print(f"[SPU] 학습 완료 — 총 {time.perf_counter()-t_train_start:.1f}s")
    sf.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="shhs/datasets/shhs1-dataset-0.21.0.csv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--dp-sigma", type=float, default=0.01)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss", choices=ALL_LOSS_CHOICES, default="mse")
    p.add_argument("--reveal-loss", action="store_true",
                    help="mse_floor_dp에서도 디버깅용으로 loss를 공개한다 (기본은 비공개)")
    a = p.parse_args()
    run(a.csv, a.epochs, a.batch_size, a.max_batches, a.dp_sigma, a.lr, a.eval_every, a.seed, a.loss, a.reveal_loss)
