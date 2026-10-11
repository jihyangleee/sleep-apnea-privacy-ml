"""현재 확정된 14개 변형 조합(times34p 포함, sleep_latency 제외)에 대해
'깨끗한 PSG 학습 -> 워치 노이즈 낀 테스트' 시나리오를 평가한다. (독립 실행 스크립트)

학습은 main.py --mode distributed --model linear 와 같은 조건
(LinearHospitalModel, Beaver sigmoid, DP sigma=0.01, 30 epochs, batch 32)을 쓴다.
학습은 1회만 깨끗한 데이터로 하고, 평가만 clean/noisy 두 갈래로 나눈다 (재학습 없음).

노이즈 모델 (raw 단위 bias/std, simulate_watch_domain_gap.py와 동일 출처):
  avgsat   : Cho et al., PMC11367728 (Galaxy Watch 4 야간 SpO2 vs PSG)
  avg_hr   : PMC9731465
  slpprdp / slpeffp / waso : JCSM 메타분석, PMC11874098
  times34p : 거친 추정치 (N3 판정은 액티그래피로 신뢰도가 특히 낮음)
  sleep_latency : 이번 조합엔 없음 (제외)
  odi / mean_desat_duration / mean_desat_drop / age_s1 / gender / bmi_s1 /
  neck20 / ess_s1 : 노이즈 모델 없음 -> 깨끗한 값 그대로 (한계로 명시)

사용법: PYTHONPATH=<프로젝트 루트> python experiments/features/watch_scenario_variant.py [csv] [n_seeds] [n_noise]
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

from dataset import (
    SHHS_FEATURES, SHHS_LABEL, _SHHS_HR_COLS, MODEL_FEATURES, DEFAULT_XML_DIR,
    _load_desat_features,
)
from model import LinearHospitalModel
from secret_sharing import additive_split, BeaverProvider

NUM_CLIENTS = 3
AHI_THRESHOLD = 15

ROLE_FEATURES = {
    "A": ["avgsat", "avg_hr", "odi", "mean_desat_duration", "mean_desat_drop"],
    "B": ["slpprdp", "slpeffp", "times34p", "waso"],
    "C": ["age_s1", "gender", "bmi_s1", "neck20", "ess_s1"],
}
FEATURES = [f for role in ROLE_FEATURES.values() for f in role]
GROUPS, _start = [], 0
for _feats in ROLE_FEATURES.values():
    GROUPS.append(list(range(_start, _start + len(_feats))))
    _start += len(_feats)

NOISE_MODEL = {  # (bias, std, lo, hi) raw 단위
    "avgsat":   (-0.2,   2.3,  70.0, 100.0),
    "avg_hr":   (0.0,    3.0,  30.0, 200.0),
    "slpprdp":  (-16.85, 30.0,  0.0, 700.0),
    "slpeffp":  (-4.69,   8.0,  0.0, 100.0),
    "waso":     (13.255, 20.0,  0.0, 400.0),
    "times34p": (0.0,     8.0,  0.0,  60.0),
}
NOISE_COLS = {FEATURES.index(name): p for name, p in NOISE_MODEL.items() if name in FEATURES}


def load_raw_and_scaled(csv_path):
    csv_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL, "nsrrid"]
    df = pd.read_csv(csv_path, usecols=csv_cols)
    df["avg_hr"] = df[_SHHS_HR_COLS].mean(axis=1)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna(subset=SHHS_FEATURES + [SHHS_LABEL])
    df = df.set_index("nsrrid").join(_load_desat_features(DEFAULT_XML_DIR), how="inner")

    col_idx = [MODEL_FEATURES.index(f) for f in FEATURES]
    X15_raw = df[MODEL_FEATURES].values.astype(np.float32)
    ahi = df[SHHS_LABEL].values.astype(np.float32)

    scaler_full = StandardScaler().fit(X15_raw)
    mean14, scale14 = scaler_full.mean_[col_idx], scaler_full.scale_[col_idx]
    X_raw = X15_raw[:, col_idx]
    X_scaled = (X_raw - mean14) / scale14
    return X_raw, X_scaled, ahi, mean14, scale14


def apply_watch_noise(X_raw, rng):
    X_noisy = X_raw.copy()
    for col, (bias, std, lo, hi) in NOISE_COLS.items():
        noise = rng.normal(bias, std, size=X_noisy.shape[0])
        X_noisy[:, col] = np.clip(X_noisy[:, col] + noise, lo, hi)
    return X_noisy


def train(X_train, y_train, feature_groups, n_epochs=30, batch_size=32, dp_sigma=0.01, seed=0):
    torch.manual_seed(seed)
    hospitals = [LinearHospitalModel(i, feature_groups) for i in range(NUM_CLIENTS)]
    X_trains = [torch.tensor(X_train[:, g], dtype=torch.float32) for g in feature_groups]
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = len(y_t)

    params = [h.sub.weight for h in hospitals] + [hospitals[0].bias]
    optimizer = torch.optim.Adam(params, lr=1e-3)
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
            triple1 = beaver.generate_triple(logit_shares[0].shape)
            triple2 = beaver.generate_triple(logit_shares[0].shape)
            pred_shares = beaver.sigmoid_approx(logit_shares, triple1, triple2)

            y_shares = additive_split(y_batch.unsqueeze(1), n=NUM_CLIENTS)
            diff = sum(pred_shares[i] - y_shares[i] for i in range(NUM_CLIENTS))
            loss = (diff ** 2).mean()
            loss.backward()
            optimizer.step()
    return hospitals


def predict_logit(hospitals, X_test, feature_groups):
    X_tests = [torch.tensor(X_test[:, g], dtype=torch.float32) for g in feature_groups]
    with torch.no_grad():
        logit = sum(hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)) + hospitals[0].bias
    return logit.squeeze().numpy()


def acc_of(y_true, logit):
    y_pred = (logit > 0).astype(np.float32)
    return float((y_pred == y_true).mean())


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    n_train_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_noise_draws = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    X_raw, X_scaled, ahi, mean14, scale14 = load_raw_and_scaled(csv_path)
    y = (ahi >= AHI_THRESHOLD).astype(np.float32)

    print(f"[data] n={len(X_raw)}, 피처 {len(FEATURES)}개: {FEATURES}")
    print(f"[noise] 적용 대상: {[FEATURES[i] for i in NOISE_COLS]}")
    print(f"[clean] 노이즈 없는 피처: {[f for i, f in enumerate(FEATURES) if i not in NOISE_COLS]}\n")

    idx = np.arange(len(X_raw))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    y_train, y_test = y[idx_train], y[idx_test]
    X_train_scaled = X_scaled[idx_train]
    X_test_raw_clean = X_raw[idx_test]

    print(f"=== 변형 조합(14개) 최종 워치 시나리오: {n_train_seeds} 학습 시드 x 노이즈 {n_noise_draws}회 ===\n")

    clean_accs, noisy_accs = [], []
    for seed in range(n_train_seeds):
        t0 = time.perf_counter()
        hospitals = train(X_train_scaled, y_train, GROUPS, seed=seed)

        X_test_clean_scaled = (X_test_raw_clean - mean14) / scale14
        logit_clean = predict_logit(hospitals, X_test_clean_scaled, GROUPS)
        a_clean = acc_of(y_test, logit_clean)
        clean_accs.append(a_clean)

        rng = np.random.default_rng(2000 + seed)
        seed_noisy = []
        for _ in range(n_noise_draws):
            X_noisy_raw = apply_watch_noise(X_test_raw_clean, rng)
            X_noisy_scaled = (X_noisy_raw - mean14) / scale14
            logit_noisy = predict_logit(hospitals, X_noisy_scaled, GROUPS)
            a_noisy = acc_of(y_test, logit_noisy)
            seed_noisy.append(a_noisy)
            noisy_accs.append(a_noisy)

        wall = time.perf_counter() - t0
        print(f"  seed {seed}: clean_acc={a_clean:.4f} | "
              f"noisy_acc={[f'{a:.4f}' for a in seed_noisy]} | {wall:.1f}s")

    clean_accs, noisy_accs = np.array(clean_accs), np.array(noisy_accs)
    print(f"\n=== 요약 ===")
    print(f"  clean_acc ({n_train_seeds} seeds)            = {clean_accs.mean():.4f} +/- {clean_accs.std():.4f}")
    print(f"  noisy_acc ({len(noisy_accs)} 학습x노이즈 조합) = {noisy_accs.mean():.4f} +/- {noisy_accs.std():.4f}")
    print(f"  domain-gap drop = {clean_accs.mean() - noisy_accs.mean():.4f}")
