import random
import torch
import torch.nn as nn

from dataset import (
    load_shhs_data, load_dreamt_data, generate_dummy_data,
    partition_data_vertical,
)
from model import HospitalModel
from secret_sharing import additive_split, apply_dp_noise, BeaverProvider

NUM_CLIENTS = 3

# Feature partition by inspection panel (index into SHHS_FEATURES)
# SHHS_FEATURES = [avgsao2, avg_hr, slptime, slp_eff, timest34p, age_s1, gender, bmi_s1]
SLEEP_FEATURE_GROUPS = [
    [0, 1],      # client A — 심박·산소 모니터링: SpO2(avgsao2), avg HR(avg_hr)
    [2, 3, 4],   # client B — 수면검사실(PSG): total sleep, efficiency, deep ratio
    [5, 6, 7],   # client C — 병원/클리닉: age, sex, BMI
]

SLEEP_CLIENT_LABELS = [
    "심박·산소 모니터링(SpO2, avg HR)",
    "수면검사실(total sleep, efficiency, deep ratio)",
    "병원(age, sex, BMI) + label",
]


def run_distributed_simulation(
    csv_path: str = None,
    dreamt_dir: str = None,
    n_epochs: int = 30,
    batch_size: int = 32,
    dp_sigma: float = 0.01,
    emb_dim: int = 16,
):
    """Fully distributed VFL training — no central server sees any embedding.

    Privacy model
    -------------
    - Sub-model runs locally on each hospital (private features never shared).
    - Embeddings are additively secret-shared before transmission.
    - DP noise injected into each share in transit.
    - Top-model is a single linear layer → decomposes over additive shares
      without any Beaver Triple (linearity eliminates the need for MPC).

    Returns: (hospitals, shared_W, scaler)
    """
    # ── Data loading ──────────────────────────────────────────────────────────
    if dreamt_dir is not None:
        print("[Distributed FL] DREAMT 데이터 로드")
        X, y, scaler = load_dreamt_data(dreamt_dir, return_scaler=True)
    elif csv_path is not None:
        print("[Distributed FL] SHHS 데이터 로드")
        X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    else:
        print("[Distributed FL] CSV not provided - synthetic dummy data 사용")
        X, y, scaler = generate_dummy_data(return_scaler=True)

    partitions, _, _, _, _ = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )

    print("[Distributed FL] feature 분배:")
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  hospital {i}: {label}")
    print(f"  DP sigma={dp_sigma}  (embedding share 전송 시 Gaussian noise)")
    print(f"  Top-model: 단일 linear — SS 분해로 Beaver Triple 없이 분산 계산")

    # ── Build hospitals with SHARED single top-model ──────────────────────────
    total_emb = emb_dim * NUM_CLIENTS
    shared_W  = nn.Linear(total_emb, 1)

    hospitals = [
        HospitalModel(i, SLEEP_FEATURE_GROUPS, emb_dim, shared_W)
        for i in range(NUM_CLIENTS)
    ]

    # ── Data tensors per hospital ─────────────────────────────────────────────
    X_trains = [
        torch.tensor(partitions[i]["X_train"], dtype=torch.float32)
        for i in range(NUM_CLIENTS)
    ]
    X_tests = [
        torch.tensor(partitions[i]["X_test"], dtype=torch.float32)
        for i in range(NUM_CLIENTS)
    ]
    y_train = torch.tensor(partitions[0]["y_train"], dtype=torch.float32)
    y_test  = torch.tensor(partitions[0]["y_test"],  dtype=torch.float32)
    n = len(y_train)

    # ── Optimizer: all sub-models + shared top linear ─────────────────────────
    params = []
    for h in hospitals:
        params.extend(h.sub.parameters())
    params.extend(shared_W.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.BCELoss()
    beaver    = BeaverProvider(NUM_CLIENTS)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[Distributed FL] 학습 시작 - {n_epochs} epochs, batch={batch_size}")
    for epoch in range(1, n_epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            idx     = perm[start : start + batch_size]
            y_batch = y_train[idx]
            optimizer.zero_grad()

            # Step 1: each hospital computes its local embedding (private)
            local_embs = [
                hospitals[i].local_emb(X_trains[i][idx]) for i in range(NUM_CLIENTS)
            ]

            # DP on gradients: prevents label inference from gradient sign/magnitude
            if dp_sigma > 0:
                for emb in local_embs:
                    emb.register_hook(lambda g: g + torch.randn_like(g) * dp_sigma)

            # Step 2: additive SS split
            all_shares = [additive_split(emb, n=NUM_CLIENTS) for emb in local_embs]

            # Step 3: DP noise in transit → each hospital receives concat of noisy shares
            concat_shares = []
            for j in range(NUM_CLIENTS):
                received = [
                    apply_dp_noise(all_shares[i][j], dp_sigma)
                    for i in range(NUM_CLIENTS)
                ]
                concat_shares.append(torch.cat(received, dim=1))
            # concat_shares[j]: (batch, NUM_CLIENTS * emb_dim) + DP noise

            # Step 4: each hospital applies shared top linear to its concat share
            # sum_j logit_share_j = W @ cat(embs) + b = logit  [by linearity]
            logit_shares = [
                hospitals[j].logit_share(concat_shares[j], j == 0)
                for j in range(NUM_CLIENTS)
            ]

            # Step 5: Beaver Triple sigmoid — σ(logit) ≈ 0.5 + 0.197x - 0.004x³
            # logit stays in share form; no party reconstructs the raw logit.
            # x³ needs 2 multiplications → 2 triples.
            triple1 = beaver.generate_triple(logit_shares[0].shape)
            triple2 = beaver.generate_triple(logit_shares[0].shape)
            pred_shares = beaver.sigmoid_approx(logit_shares, triple1, triple2)

            # Label holder sums pred shares → sees only the final prediction [0,1]
            pred = sum(pred_shares).clamp(1e-6, 1 - 1e-6)
            loss = criterion(pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                test_embs = torch.cat(
                    [hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)],
                    dim=1,
                )
                logit = shared_W(test_embs)
                pred  = (logit.squeeze() > 0).float()
                acc   = (pred == y_test).float().mean().item()
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"loss={epoch_loss/n_batches:.4f} | test_acc={acc:.4f}"
            )

    print("[Distributed FL] 학습 완료")
    return hospitals, shared_W, scaler
