"""MPC 안에서 비교 연산 없이 BCE를 쓰는 방법 검증.

아이디어: z = sum_i(w_i . x_i) + b 에서, 각 병원이 로컬 평문으로
  - x_i의 각 성분을 clip_x로 clip (표준화된 피처의 outlier 제거, 흔한 전처리)
  - 학습 스텝마다 ||w_i||_2를 w_max_i 이하로 projection
을 하면, 코시-슈바르츠 부등식에 의해
  |z| <= sum_i ||w_i||_2 * ||x_i||_2 + |b| <= sum_i w_max_i * x_norm_max_i + b_max
가 수학적으로 "항상" 보장된다. 이 전처리/projection은 전부 각 병원의 로컬 평문
연산이라 MPC 비용이 0이고, SPU 안에서는 z가 이미 안전한 범위라 clip(비교)이
필요 없다 -- bce_logit_trick 그래디언트 (pred-y)/B를 그대로 써도 된다.

이 스크립트는 target z-bound를 몇 가지로 바꿔가며, 그 보장을 지키기 위해
가중치를 얼마나 세게 눌러야 하는지와 그 결과 정확도가 어떻게 되는지를 잰다.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import SHHS_FEATURES, SHHS_LABEL, _SHHS_HR_COLS, MODEL_FEATURES, DEFAULT_XML_DIR, _load_desat_features

AHI_THRESHOLD = 15
ROLE_FEATURES = {
    "A": ["avgsat", "avg_hr", "odi", "mean_desat_duration", "mean_desat_drop"],
    "B": ["slpprdp", "slpeffp", "times34p", "waso"],
    "C": ["age_s1", "gender", "bmi_s1", "neck20", "ess_s1"],
}
FEATURES = [f for role in ROLE_FEATURES.values() for f in role]
GROUP_SIZES = [len(v) for v in ROLE_FEATURES.values()]  # [5, 4, 5]


def sigmoid_approx(z):
    return 0.5 + 0.197 * z - 0.004 * z ** 3


def load_data(csv_path):
    csv_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL, "nsrrid"]
    df = pd.read_csv(csv_path, usecols=csv_cols)
    df["avg_hr"] = df[_SHHS_HR_COLS].mean(axis=1)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna(subset=SHHS_FEATURES + [SHHS_LABEL])
    df = df.set_index("nsrrid").join(_load_desat_features(DEFAULT_XML_DIR), how="inner")

    col_idx = [MODEL_FEATURES.index(f) for f in FEATURES]
    X = df[MODEL_FEATURES].values.astype(np.float32)[:, col_idx]
    y = (df[SHHS_LABEL].values >= AHI_THRESHOLD).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    return X, y


def train_bounded(X_train, y_train, loss_type, target_z_bound, clip_x, n_epochs=30, batch_size=32, lr=1e-3, seed=0,
                   unconstrained=False):
    """loss_type: 'mse' | 'bce_free' (clip 없는 (pred-y)/B, z는 가중치 projection으로만 제한).
    unconstrained=True 면 clip도 projection도 전혀 안 함 (순수 기준선)."""
    torch.manual_seed(seed)

    # 피처를 그룹별로 분리 (각 병원이 들고 있는 열)
    offsets = np.cumsum([0] + GROUP_SIZES)
    X_groups = [X_train[:, offsets[i]:offsets[i + 1]] for i in range(3)]
    if not unconstrained:
        X_groups = [np.clip(g, -clip_x, clip_x) for g in X_groups]  # 로컬 평문 clip (outlier 제거, MPC 비용 0)

    if unconstrained:
        w_max = [1e18, 1e18, 1e18]
        b_budget = 1e18
        x_norm_max = [None, None, None]
    else:
        x_norm_max = [clip_x * np.sqrt(n) for n in GROUP_SIZES]  # 각 그룹의 ||x_i|| 최대 가능치
        b_budget = 0.2
        w_budget_total = max(target_z_bound - b_budget, 0.3)
        w_max = [w_budget_total / 3 / xn for xn in x_norm_max]  # 그룹마다 균등 배분

    ws = [torch.zeros(n, 1, requires_grad=True) for n in GROUP_SIZES]
    for w in ws:
        torch.nn.init.normal_(w, std=0.01)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam(ws + [b], lr=lr)

    X_t = [torch.tensor(g, dtype=torch.float32) for g in X_groups]
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    n = len(y_t)

    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            y_b = y_t[idx]
            optimizer.zero_grad()

            s = [X_t[i][idx] @ ws[i] for i in range(3)]  # 각 병원의 로컬 s_i = w_i . x_i
            z = sum(s) + b.clamp(-b_budget, b_budget)
            pred = sigmoid_approx(z)

            if loss_type == "mse":
                loss = ((pred - y_b) ** 2).mean()
                loss.backward()
            elif loss_type == "bce_free":
                grad_z = (pred.detach() - y_b) / idx.shape[0]  # clip 없이 그대로 주입
                z.backward(gradient=grad_z)
            optimizer.step()

            # 각 병원이 로컬로 자기 가중치 norm을 projection (MPC 비용 0)
            with torch.no_grad():
                for i, w in enumerate(ws):
                    nrm = w.norm()
                    if nrm > w_max[i]:
                        w.mul_(w_max[i] / nrm)

    return ws, b, x_norm_max, w_max


def evaluate(ws, b, X_test, y_test):
    offsets = np.cumsum([0] + GROUP_SIZES)
    X_groups = [X_test[:, offsets[i]:offsets[i + 1]] for i in range(3)]
    with torch.no_grad():
        z = sum(torch.tensor(g, dtype=torch.float32) @ ws[i] for i, g in enumerate(X_groups)) + b
        z = z.squeeze().numpy()
    y_pred = (z > 0).astype(np.float32)
    acc = float((y_pred == y_test).mean())
    auc = roc_auc_score(y_test, z)
    return acc, auc, float(np.abs(z).max())


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    X, y = load_data(csv_path)
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"[data] n={len(X)}, 피처 {len(FEATURES)}개 (그룹 크기 {GROUP_SIZES})\n")

    configs = [
        ("mse_unconstrained",        "mse",      None, None, True),
        ("bce_free_bound4_clipx3",   "bce_free", 4.0, 3.0, False),
        ("bce_free_bound7_clipx3",   "bce_free", 7.0, 3.0, False),
        ("bce_free_bound10_clipx3",  "bce_free", 10.0, 3.0, False),
        ("bce_free_bound15_clipx3",  "bce_free", 15.0, 3.0, False),
        ("bce_free_bound20_clipx3",  "bce_free", 20.0, 3.0, False),
        ("bce_free_bound30_clipx3",  "bce_free", 30.0, 3.0, False),
    ]

    print("=== 비교 연산 없는 BCE (가중치 projection으로 z를 제한) ===\n")
    results = {}
    for name, loss_type, target_bound, clip_x, unconstrained in configs:
        t0 = time.perf_counter()
        accs, aucs, maxzs = [], [], []
        w_max_report = None
        for seed in range(n_seeds):
            ws, b, xnm, wmax = train_bounded(X_train, y_train, loss_type, target_bound, clip_x, seed=seed,
                                              unconstrained=unconstrained)
            w_max_report = wmax
            acc, auc, maxz = evaluate(ws, b, X_test, y_test)
            accs.append(acc); aucs.append(auc); maxzs.append(maxz)
        wall = time.perf_counter() - t0
        accs, aucs, maxzs = np.array(accs), np.array(aucs), np.array(maxzs)
        results[name] = accs
        print(f"  {name:28s}: acc={accs.mean():.4f}+/-{accs.std():.4f}  auc={aucs.mean():.4f}+/-{aucs.std():.4f}  "
              f"|z|max_observed={maxzs.max():.2f}  w_max={[f'{w:.3f}' for w in w_max_report] if w_max_report else '-'}  ({wall:.1f}s)")

    base = results["mse_unconstrained"].mean()
    print(f"\n=== MSE 대비 ===")
    for name in results:
        if name == "mse_unconstrained":
            continue
        diff = results[name].mean() - base
        print(f"  {name:28s}: {diff:+.4f} ({diff*100:+.2f}pp)")
