"""평문 추론 vs CKKS(HE) 추론의 분류 정확도가 실제로 같은지, 테스트셋 샘플로 직접 측정한다.

지금까지 he-infer 데모(main.py --mode he-infer)는 더미 프로필 5개로만 |err|를
확인했다 (논문 표8의 "측정 필요" 항목과 동일한 공백). 여기서는 실제 SHHS 테스트셋
샘플에 대해 같은 모델로 (a) 평문 추론 (b) CKKS 추론을 둘 다 돌려서
  - 두 추론의 예측 클래스가 얼마나 일치하는지
  - logit 오차(|z_CKKS - z_plain|)의 평균/최대
  - 각각을 실제 라벨과 비교했을 때 정확도가 같은지
를 측정한다. CKKS 연산이 샘플당 비용이 있어 전체 테스트셋이 아니라 N개 서브샘플로
돈다 (N은 인자로 조절).

사용법: PYTHONPATH=<프로젝트 루트> python experiments/features/he_vs_plaintext_accuracy.py [csv] [N]
"""

import sys
import time

import numpy as np
import torch
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import load_shhs_data
from model import LinearHospitalModel
from secret_sharing import additive_split, BeaverProvider
from simulate import SLEEP_FEATURE_GROUPS, SLEEP_CLIENT_LABELS
from he_client import build_he_context

NUM_CLIENTS = 3


def train_quick(X_train, y_train, n_epochs=30, batch_size=32, dp_sigma=0.01, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    hospitals = [LinearHospitalModel(i, SLEEP_FEATURE_GROUPS) for i in range(NUM_CLIENTS)]
    X_trains = [torch.tensor(X_train[:, g], dtype=torch.float32) for g in SLEEP_FEATURE_GROUPS]
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
            triple1 = beaver.generate_triple(logit_shares[0].shape)
            triple2 = beaver.generate_triple(logit_shares[0].shape)
            pred_shares = beaver.sigmoid_approx(logit_shares, triple1, triple2)
            y_shares = additive_split(y_batch.unsqueeze(1), n=NUM_CLIENTS)
            diff = sum(pred_shares[i] - y_shares[i] for i in range(NUM_CLIENTS))
            loss = (diff ** 2).mean()
            loss.backward()
            optimizer.step()
    return hospitals


def plain_logit(hospitals, x_raw_scaled):
    with torch.no_grad():
        shares = [hospitals[i].local_emb(torch.tensor(x_raw_scaled[hospitals[i].feature_groups[i]],
                                                        dtype=torch.float32).unsqueeze(0)) for i in range(NUM_CLIENTS)]
        return float(sum(shares).item() + hospitals[0].bias.item())


def he_logit(hospitals, x_raw_scaled, patient_ctx, hospital_ctx):
    all_enc_xi = {}
    for h in hospitals:
        raw = x_raw_scaled[h.feature_groups[h.id]].astype(np.float64)
        pad_to = max(1 << (len(raw) - 1).bit_length(), 4)
        padded = np.zeros(pad_to)
        padded[:len(raw)] = raw
        all_enc_xi[h.id] = ts.ckks_vector(patient_ctx, padded.tolist()).serialize()
    entry = hospitals[0]
    others = hospitals[1:]
    enc_result = entry.run_he_inference(all_enc_xi, hospital_ctx, others)
    vec = ts.lazy_ckks_vector_from(enc_result)
    vec.link_context(patient_ctx)
    return float(vec.decrypt()[0])


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"[data] train n={len(X_train)}, test n={len(X_test)}")

    print("[train] 15개 피처, linear, MSE, 30 epochs (이 비교 목적엔 손실함수 무관 -- 추론 파이프라인만 확인)")
    hospitals = train_quick(X_train, y_train, seed=0)
    for h in hospitals:
        h.eval()
    b_top = hospitals[0].bias.detach().numpy()
    for h in hospitals:
        h.build_he_weights(b_top)

    patient_ctx = build_he_context()
    hospital_ctx = ts.context_from(patient_ctx.serialize(save_secret_key=False))

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(X_test), size=min(N, len(X_test)), replace=False)

    print(f"\n[infer] {len(sample_idx)}개 테스트 샘플에 대해 평문 vs CKKS 비교 중...")
    t0 = time.perf_counter()
    plain_logits, he_logits, labels = [], [], []
    for n, i in enumerate(sample_idx, 1):
        x = X_test[i]
        zp = plain_logit(hospitals, x)
        zh = he_logit(hospitals, x, patient_ctx, hospital_ctx)
        plain_logits.append(zp); he_logits.append(zh); labels.append(y_test[i])
        if n % 10 == 0:
            print(f"  {n}/{len(sample_idx)} ... ({time.perf_counter()-t0:.1f}s 경과)")
    wall = time.perf_counter() - t0

    plain_logits = np.array(plain_logits); he_logits = np.array(he_logits); labels = np.array(labels)
    err = np.abs(he_logits - plain_logits)
    plain_pred = (plain_logits > 0).astype(np.float32)
    he_pred = (he_logits > 0).astype(np.float32)

    print(f"\n=== 결과 ({len(sample_idx)}개 샘플, 총 {wall:.1f}s, 샘플당 평균 {wall/len(sample_idx):.2f}s) ===")
    print(f"  logit 오차 |z_CKKS - z_plain|: 평균={err.mean():.8f}  최대={err.max():.8f}")
    print(f"  예측 클래스 일치율 (평문 vs CKKS): {(plain_pred == he_pred).mean():.4f}")
    print(f"  정확도 (평문 추론 vs 실제 라벨): {(plain_pred == labels).mean():.4f}")
    print(f"  정확도 (CKKS 추론 vs 실제 라벨): {(he_pred == labels).mean():.4f}")
