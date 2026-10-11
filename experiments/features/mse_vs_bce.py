"""MSE vs BCE(로짓 트릭) 비교 -- 평문, 다항식 근사 시그모이드(sigmoid_approx) 동일 사용.

목적: "BCE로 바꾸면 정확도가 얼마나 오르는가"를 추측이 아니라 직접 측정한다.
MPC/SPU 없이 평문 PyTorch로 돌리되, secret_sharing.py와 완전히 같은 다항식
sigma_approx(x) = 0.5 + 0.197x - 0.004x^3 를 그대로 쓴다.

두 경로의 차이는 "z(logit)에 주입하는 그래디언트"뿐이다:
  MSE : loss = (pred - y)^2 의 autograd 그래디언트 그대로
        (= 2*(pred-y)*sigma_approx'(z)/batch, 현재 MPC 경로와 수학적으로 동일)
  BCE : z.backward(grad_z) 로 grad_z = (pred - y)/batch 를 직접 주입
        (정확한 시그모이드였다면 나왔을 깔끔한 공식을 근사 시그모이드에 그대로 빌려씀 --
         "근사의 근사"라는 점은 지난 설명 그대로)

모델은 선형모델이라 병원 3곳 분배와 무관하게 "전체 피처에 대한 단일 로지스틱 회귀"와
동일하므로, 여기서는 14개 피처를 합쳐 하나의 Linear(14,1)로 학습한다 (결과는 동일).

사용법: PYTHONPATH=<프로젝트 루트> python experiments/features/mse_vs_bce.py [csv] [n_seeds]
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


def sigmoid_approx(z):
    # secret_sharing.py BeaverProvider.sigmoid_approx 와 동일 다항식
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


def train_one(X_train, y_train, loss_type, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    n_feat = X_train.shape[1]
    w = torch.zeros(n_feat, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    torch.nn.init.normal_(w, std=0.01)
    optimizer = torch.optim.Adam([w, b], lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    n = len(y_t)

    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x_b, y_b = X_t[idx], y_t[idx]
            optimizer.zero_grad()

            z = x_b @ w + b
            pred = sigmoid_approx(z)

            if loss_type == "mse":
                loss = ((pred - y_b) ** 2).mean()
                loss.backward()
            elif loss_type == "bce_logit_trick":
                grad_z = (pred.detach() - y_b) / x_b.shape[0]
                z.backward(gradient=grad_z)
            elif loss_type == "bce_clip_z":
                # 근본 원인(시그마틸드가 유효한 구간 밖으로 나가는 것) 자체를 막는다:
                # z를 시그모이드 근사가 그나마 타당한 구간으로 먼저 clip한 뒤 pred/grad 계산
                z_clipped = z.clamp(-4.0, 4.0)
                pred_c = sigmoid_approx(z_clipped)
                grad_z = (pred_c.detach() - y_b) / x_b.shape[0]
                z.backward(gradient=grad_z)
            elif loss_type == "bce_clip_z_32":
                # 더 보수적인 경계: |z|<3.2 (sigma_tilde가 진짜 0~1 사이인 구간)
                z_clipped = z.clamp(-3.2, 3.2)
                pred_c = sigmoid_approx(z_clipped)
                grad_z = (pred_c.detach() - y_b) / x_b.shape[0]
                z.backward(gradient=grad_z)
            elif loss_type == "bce_seg3":
                # SPU 내장 SIGMOID_SEG3 (sml/utils/fxp_approx.py sigmoid_seg3):
                # |z|<=4 이면 0.5+0.125z, 바깥은 0/1 -- 3차식 대신 1차식 + 양끝 고정
                pred_s = torch.where(z < -4, torch.zeros_like(z),
                                     torch.where(z > 4, torch.ones_like(z), 0.5 + 0.125 * z))
                grad_z = (pred_s.detach() - y_b) / x_b.shape[0]
                z.backward(gradient=grad_z)
            elif loss_type == "bce_clip_pred":
                # pred 값만 [0,1]로 clip (z는 그대로, 다항식이 0~1 밖으로 나간 값만 눌러줌)
                pred_c = pred.clamp(0.0, 1.0)
                grad_z = (pred_c.detach() - y_b) / x_b.shape[0]
                z.backward(gradient=grad_z)
            elif loss_type == "bce_clip_grad":
                # 그래디언트 자체를 고정 범위로 clip (가장 단순한 범용 안전장치)
                grad_z = ((pred.detach() - y_b) / x_b.shape[0]).clamp(-0.1, 0.1)
                z.backward(gradient=grad_z)
            elif loss_type == "bce_sigprime_floor":
                # MSE가 원래 갖던 sigma'(z) 보정을 살리되, 부호가 뒤집히는 걸 막기 위해
                # 0 밑으로는 못 내려가게 바닥(floor)을 둔다 -- "안전한 MSE/BCE 혼합"
                sigma_prime = 0.197 - 0.012 * z ** 2
                weight = sigma_prime.clamp(min=0.02)  # 완전히 0이 되는 것도 방지
                grad_z = (pred.detach() - y_b) * weight.detach() / x_b.shape[0]
                z.backward(gradient=grad_z)
            else:
                raise ValueError(loss_type)

            optimizer.step()

    return w.detach(), b.detach()


def evaluate(w, b, X_test, y_test):
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        z = (X_t @ w + b).squeeze().numpy()
    y_pred = (z > 0).astype(np.float32)
    acc = float((y_pred == y_test).mean())
    auc = roc_auc_score(y_test, z)
    return acc, auc


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    X, y = load_data(csv_path)
    print(f"[data] n={len(X)}, 피처 {len(FEATURES)}개: {FEATURES}")
    print(f"[data] AHI>={AHI_THRESHOLD} 양성 비율: {y.mean():.3f}\n")

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"=== MSE vs BCE 변형들, 동일 sigmoid_approx, {n_seeds} seeds ===\n")
    loss_types = ["mse", "bce_logit_trick", "bce_clip_z", "bce_clip_z_32", "bce_seg3", "bce_clip_pred", "bce_clip_grad", "bce_sigprime_floor"]
    results = {lt: [] for lt in loss_types}
    for loss_type in loss_types:
        t0 = time.perf_counter()
        for seed in range(n_seeds):
            w, b = train_one(X_train, y_train, loss_type, seed=seed)
            acc, auc = evaluate(w, b, X_test, y_test)
            results[loss_type].append((acc, auc))
        wall = time.perf_counter() - t0
        accs = np.array([r[0] for r in results[loss_type]])
        aucs = np.array([r[1] for r in results[loss_type]])
        print(f"  {loss_type:18s}: acc={accs.mean():.4f}+/-{accs.std():.4f}  "
              f"auc={aucs.mean():.4f}+/-{aucs.std():.4f}  ({wall:.1f}s, {n_seeds} seeds)")

    mse_acc = np.array([r[0] for r in results["mse"]])
    print(f"\n=== MSE 대비 차이 ===")
    for loss_type in loss_types[1:]:
        acc = np.array([r[0] for r in results[loss_type]])
        diff = acc.mean() - mse_acc.mean()
        print(f"  {loss_type:20s}: {diff:+.4f} ({diff*100:+.2f}pp)")
