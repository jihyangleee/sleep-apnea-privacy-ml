"""논문 Table 4와 동일한 원래 15개 피처(times34p, sleep_latency 포함) 구성으로,
MSE 대신 BCE-safe(sigma'(z) floor) 그래디언트를 쓴 '자체 시뮬레이션' 결과를 측정한다.

simulate.py의 SLEEP_FEATURE_GROUPS(논문 표4와 동일 5:5:5 분할)를 그대로 쓰고,
학습 루프는 LinearHospitalModel + BeaverProvider(시그모이드까지는 동일) 구조를
유지하되, 역전파에 주입하는 그래디언트만 MSE의 2*(pred-y)*sigma'(z)/B 대신
(pred-y)*max(sigma'(z), floor)/B 로 바꾼다 (mse_vs_bce.py의 bce_sigprime_floor와
동일 공식, 최고 성능이었던 조합).

사용법: PYTHONPATH=<프로젝트 루트> python experiments/features/bce_safe_15features_selfsim.py [csv] [n_seeds]
"""

import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import load_shhs_data, MODEL_FEATURES
from model import LinearHospitalModel
from secret_sharing import additive_split, BeaverProvider
from simulate import SLEEP_FEATURE_GROUPS, SLEEP_CLIENT_LABELS

NUM_CLIENTS = 3


def train(X_train, y_train, feature_groups, loss_type, n_epochs=30, batch_size=32, dp_sigma=0.01, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    hospitals = [LinearHospitalModel(i, feature_groups) for i in range(NUM_CLIENTS)]
    X_trains = [torch.tensor(X_train[:, g], dtype=torch.float32) for g in feature_groups]
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = len(y_t)

    params = [h.sub.weight for h in hospitals] + [hospitals[0].bias]
    optimizer = torch.optim.Adam(params, lr=lr)
    beaver = BeaverProvider(NUM_CLIENTS)

    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            y_batch = y_t[idx]
            optimizer.zero_grad()

            local_embs = [hospitals[i].local_emb(X_trains[i][idx]) for i in range(NUM_CLIENTS)]
            if dp_sigma > 0:
                for emb in local_embs:
                    emb.register_hook(lambda g: g + torch.randn_like(g) * dp_sigma)

            logit_shares = [hospitals[j].logit_share(local_embs[j], j == 0) for j in range(NUM_CLIENTS)]
            z = sum(logit_shares)  # 비밀분산 환경에서는 복원되지 않는 값 -- 여기선 그래디언트 공식 계산용으로만 평문 사용
            pred = 0.5 + 0.197 * z - 0.004 * z ** 3

            y_shares = additive_split(y_batch.unsqueeze(1), n=NUM_CLIENTS)
            y_sum = sum(y_shares)

            if loss_type == "mse":
                triple1 = beaver.generate_triple(logit_shares[0].shape)
                triple2 = beaver.generate_triple(logit_shares[0].shape)
                pred_shares = beaver.sigmoid_approx(logit_shares, triple1, triple2)
                diff = sum(pred_shares[i] - y_shares[i] for i in range(NUM_CLIENTS))
                loss = (diff ** 2).mean()
                loss.backward()
            elif loss_type == "bce_safe":
                sigma_prime = 0.197 - 0.012 * z ** 2
                weight = sigma_prime.clamp(min=0.02)
                grad_z = (pred.detach() - y_sum) * weight.detach() / idx.shape[0]
                for ls in logit_shares:
                    ls.backward(gradient=grad_z)  # 모든 병원이 같은 grad_z를 받는다 (블록-선형 구조)
            elif loss_type == "mse_floor":
                # MSE 공식 그대로(계수 2 유지) + sigma'(z)에만 바닥(floor)을 둔 버전.
                # "손실을 바꾼다"가 아니라 "MSE 그래디언트의 안전장치만 추가한다"는 관점.
                sigma_prime = 0.197 - 0.012 * z ** 2
                weight = sigma_prime.clamp(min=0.02)
                grad_z = 2 * (pred.detach() - y_sum) * weight.detach() / idx.shape[0]
                for ls in logit_shares:
                    ls.backward(gradient=grad_z)
            optimizer.step()

    return hospitals


def evaluate(hospitals, X_test, y_test, feature_groups):
    X_tests = [torch.tensor(X_test[:, g], dtype=torch.float32) for g in feature_groups]
    with torch.no_grad():
        logit = sum(hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)) + hospitals[0].bias
    z = logit.squeeze().numpy()
    y_pred = (z > 0).astype(np.float32)
    acc = float((y_pred == y_test).mean())
    auc = roc_auc_score(y_test, z)
    tp = np.sum((y_pred == 1) & (y_test == 1)); fn = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0)); fp = np.sum((y_pred == 1) & (y_test == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return acc, auc, sens, spec


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    print(f"[data] n={len(X)}, 피처 {X.shape[1]}개 (논문 표4 구성, 5:5:5 분할)")
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  hospital {i}: {label}")

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"\n=== 자체 시뮬레이션: MSE vs BCE-safe, 15개 피처(논문 구성), {n_seeds} seeds ===\n")
    loss_types = ["mse", "bce_safe", "mse_floor"]
    results = {lt: [] for lt in loss_types}
    for loss_type in loss_types:
        t0 = time.perf_counter()
        for seed in range(n_seeds):
            hospitals = train(X_train, y_train, SLEEP_FEATURE_GROUPS, loss_type, seed=seed)
            acc, auc, sens, spec = evaluate(hospitals, X_test, y_test, SLEEP_FEATURE_GROUPS)
            results[loss_type].append((acc, auc, sens, spec))
        wall = time.perf_counter() - t0
        arr = np.array(results[loss_type])
        print(f"  {loss_type:10s}: acc={arr[:,0].mean():.4f}+/-{arr[:,0].std():.4f}  "
              f"auc={arr[:,1].mean():.4f}+/-{arr[:,1].std():.4f}  "
              f"sens={arr[:,2].mean():.4f}+/-{arr[:,2].std():.4f}  "
              f"spec={arr[:,3].mean():.4f}+/-{arr[:,3].std():.4f}  ({wall:.1f}s, {n_seeds} seeds)")

    mse_arr = np.array(results["mse"])
    print(f"\n=== MSE 대비 ===")
    for lt in loss_types[1:]:
        arr = np.array(results[lt])
        d_acc = arr[:,0].mean() - mse_arr[:,0].mean()
        d_auc = arr[:,1].mean() - mse_arr[:,1].mean()
        print(f"  {lt:12s}: acc {d_acc:+.4f} ({d_acc*100:+.2f}pp)   auc {d_auc:+.4f}")
