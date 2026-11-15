"""DP 노이즈 σ sweep: 정확도 vs 라벨 추론 공격 vs (대략적인) ε.

train_spu_party.py(bce_clip_z_dp)의 학습 수식을 평문 numpy로 그대로 재현한다.
  - 같은 데이터(mpc/hospital_data/{A,B,C}.csv), 같은 라벨(ahi_a0h3a >= 15)
  - 같은 분할(KFold 5, shuffle, random_state=42), 학습 행으로만 표준화
  - 같은 초기화(seed*100+i), 같은 배치 순서(default_rng(seed)), 마지막 부분 배치 버림
  - grad_z = (sigma_hat(clip(z,-4,4)) - y)/B + N(0, σ²)   (세 조각 N(0,σ²/3)의 합과 같은 분포)
  - 병원별 Adam (torch 기본값), lr 1e-2, batch 256, 100 epoch
SPU와의 차이는 고정소수점 연산 오차뿐이다. σ=0.01 결과를 SPU CV 로그
(mpc/logs/cv/summary.txt)와 비교해 재현이 맞는지 확인할 수 있다.

라벨 추론 공격 (라벨 없는 병원 A의 관점, 공개받은 grad_z만 사용):
  - per-step : 각 스텝에서 받은 grad_z 하나로 라벨 추정 (-grad 점수의 AUC)
  - 누적     : 같은 환자에 대해 전 epoch 동안 받은 grad_z를 평균해 추정 (AUC)
  AUC 0.5 = 추론 불가, 1.0 = 완전 노출.

ε (label-DP, 대략): 라벨 하나가 바뀌면 그 환자의 grad_z가 1/B 바뀐다 (민감도 Δ=1/B).
  epoch마다 한 번씩 공개되므로 Gaussian mechanism을 epochs번 합성 (zCDP):
  ρ = epochs·Δ²/(2σ²),  ε = ρ + 2·sqrt(ρ·ln(1/δ)),  δ = 1e-5.
  서브샘플링 증폭을 쓰지 않은 보수적 상한이고, 형식적 accountant(RDP 등)는 아니다.

사용법 (프로젝트 루트에서):
    python experiments/dp_sigma_sweep.py
    python experiments/dp_sigma_sweep.py --sigmas 0 0.01 0.05 --noise-seeds 1
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mpc", "hospital_data")
ROLES = ["A", "B", "C"]   # hospital0, hospital1(라벨+...), hospital2 -- cluster_local.json과 같은 순서
LABEL_COL, AHI_THRESHOLD = "ahi_a0h3a", 15
LABEL_IDX, BIAS_IDX = 1, 0  # README: 라벨은 hospital1(B), bias는 hospital0(A)


def sigmoid_approx(z):
    return 0.5 + 0.197 * z - 0.004 * z ** 3


def load():
    Xs, y = [], None
    for r in ROLES:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{r}.csv"), index_col="nsrrid")
        if LABEL_COL in df:
            if y is None:
                y = (df[LABEL_COL].values >= AHI_THRESHOLD).astype(np.float32)
            df = df.drop(columns=[LABEL_COL])
        Xs.append(df.values.astype(np.float32))
    assert len({x.shape[0] for x in Xs}) == 1
    return Xs, y


def init_state(n_feat, has_bias, seed):
    rng = np.random.default_rng(seed)
    bound = 1.0 / np.sqrt(n_feat)
    p = {"w": rng.uniform(-bound, bound, size=(n_feat, 1)).astype(np.float32)}
    if has_bias:
        p["b"] = np.zeros((1,), dtype=np.float32)
    return {"p": p, "m": {k: np.zeros_like(v) for k, v in p.items()},
            "v": {k: np.zeros_like(v) for k, v in p.items()}, "t": 0}


def adam_step(s, Xb, g, lr, b1=0.9, b2=0.999, eps=1e-8):
    grads = {"w": Xb.T @ g}
    if "b" in s["p"]:
        grads["b"] = g.sum(axis=0)
    s["t"] += 1
    for k in s["p"]:
        s["m"][k] = b1 * s["m"][k] + (1 - b1) * grads[k]
        s["v"][k] = b2 * s["v"][k] + (1 - b2) * grads[k] ** 2
        mh = s["m"][k] / (1 - b1 ** s["t"])
        vh = s["v"][k] / (1 - b2 ** s["t"])
        s["p"][k] = (s["p"][k] - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)


def logit(states, X, idx):
    z = sum(X[i][idx] @ states[i]["p"]["w"] for i in range(len(X)))
    return z + states[BIAS_IDX]["p"]["b"]


def run_one(X_raw, y, idx_train, idx_test, sigma, epochs, batch, lr, seed, noise_seed):
    X = []
    for x in X_raw:
        sc = StandardScaler().fit(x[idx_train])
        X.append(sc.transform(x).astype(np.float32))
    states = [init_state(X[i].shape[1], i == BIAS_IDX, seed * 100 + i) for i in range(len(X))]
    shuffle_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(noise_seed)

    n = len(y)
    g_sum = np.zeros(n)          # 공격자(병원 A)가 환자별로 누적한 공개 grad_z
    g_cnt = np.zeros(n)
    step_scores, step_labels = [], []

    for _ in range(epochs):
        perm = idx_train[shuffle_rng.permutation(len(idx_train))]
        for start in range(0, len(perm) - batch + 1, batch):
            idx = perm[start:start + batch]
            z = logit(states, X, idx)
            yb = y[idx][:, None]
            g = (sigmoid_approx(np.clip(z, -4.0, 4.0)) - yb) / batch
            if sigma > 0:
                g = g + noise_rng.normal(0.0, sigma, size=g.shape).astype(np.float32)
            for i in range(len(X)):
                adam_step(states[i], X[i][idx], g, lr)
            g_sum[idx] += g[:, 0]
            g_cnt[idx] += 1
            step_scores.append(-g[:, 0])
            step_labels.append(y[idx])

    acc = float(np.mean((logit(states, X, idx_test)[:, 0] > 0) == (y[idx_test] > 0.5)))
    seen = g_cnt > 0
    auc_step = roc_auc_score(np.concatenate(step_labels), np.concatenate(step_scores))
    auc_cum = roc_auc_score(y[seen], -(g_sum[seen] / g_cnt[seen]))
    return acc, auc_step, auc_cum


def approx_epsilon(sigma, epochs, batch, delta=1e-5):
    if sigma <= 0:
        return float("inf")
    rho = epochs * (1.0 / batch) ** 2 / (2 * sigma ** 2)
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--noise-seeds", type=int, default=3, help="σ·fold마다 노이즈 시드 반복 횟수")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "mpc", "logs", "dp_sigma_sweep.csv"))
    a = ap.parse_args()

    X_raw, y = load()
    n = len(y)
    print(f"[data] n={n}, 양성 {y.mean():.1%}, 피처 {[x.shape[1] for x in X_raw]}", flush=True)
    splits = list(KFold(n_splits=a.n_folds, shuffle=True, random_state=42).split(np.arange(n)))

    rows = []
    t0 = time.perf_counter()
    for sigma in a.sigmas:
        for k, (tr, te) in enumerate(splits):
            for ns in range(a.noise_seeds if sigma > 0 else 1):
                acc, auc_s, auc_c = run_one(X_raw, y, tr, te, sigma, a.epochs, a.batch_size,
                                            a.lr, a.seed, noise_seed=1000 * k + ns)
                rows.append({"sigma": sigma, "fold": k, "noise_seed": ns, "test_acc": acc,
                             "attack_auc_step": auc_s, "attack_auc_cum": auc_c})
        r = pd.DataFrame([x for x in rows if x["sigma"] == sigma])
        print(f"σ={sigma:<6} acc={r.test_acc.mean():.4f}±{r.test_acc.std(ddof=0):.4f}  "
              f"attackAUC step={r.attack_auc_step.mean():.3f} cum={r.attack_auc_cum.mean():.3f}  "
              f"ε≈{approx_epsilon(sigma, a.epochs, a.batch_size):.1f}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    summ = df.groupby("sigma").agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                                   auc_step=("attack_auc_step", "mean"), auc_cum=("attack_auc_cum", "mean"))
    summ["eps_approx"] = [approx_epsilon(s, a.epochs, a.batch_size) for s in summ.index]
    summ["snr(Δ/σ)"] = [(1 / a.batch_size) / s if s > 0 else np.inf for s in summ.index]
    print("\n" + summ.round(4).to_string())
    summ.to_csv(a.out.replace(".csv", "_summary.csv"))
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
