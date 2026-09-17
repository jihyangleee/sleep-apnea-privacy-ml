import random
import time                                            # 학습 단계별 시간 오버헤드 측정용
import torch
import torch.nn as nn

from dataset import load_shhs_data, partition_data_vertical
from model import HospitalModel
from secret_sharing import additive_split, BeaverProvider

NUM_CLIENTS = 3

SLEEP_FEATURE_GROUPS = [
    [0, 1],            # client A — 심박·산소 모니터링: SpO2(avgsao2), avg HR(avg_hr)
    [2, 3, 4, 8, 9],   # client B — 수면검사실(PSG): total sleep, efficiency, deep ratio, waso, sleep_latency
    [5, 6, 7, 10, 11], # client C — 병원/클리닉: age, sex, BMI, neck20, ess_s1
]

SLEEP_CLIENT_LABELS = [
    "심박·산소 모니터링(SpO2, avg HR)",
    "수면검사실(total sleep, efficiency, deep ratio, waso, sleep_latency)",
    "병원(age, sex, BMI, neck20, ess_s1) + label",
]


def run_distributed_simulation(
    csv_path: str,
    n_epochs: int = 30,
    batch_size: int = 32,
    dp_sigma: float = 0.01,
    emb_dim: int = 16,
):
    """Fully distributed VFL training — no central server sees any embedding.

    Privacy model
    -------------
    - Sub-model runs locally on each hospital (private features never shared).
    - Embeddings never leave the hospital that computed them — each hospital
      applies its own W_top column-slice to its own local embedding directly,
      so no embedding secret-sharing or cross-hospital transmission happens.
    - The resulting per-hospital logit is already an additive share of the
      true logit (block-linearity of W_top over the concatenation); Beaver
      Triple is used only to secure the nonlinear sigmoid computed over
      these shares, so no party ever reconstructs the raw logit.
    - Labels are additively secret-shared too — no single "label holder"
      party ever sees y in the clear. Loss is MSE (not BCE) specifically
      because its gradient is a plain subtraction (pred - y), which additive
      shares support natively; BCE's gradient needs division, which would
      require a secure-division protocol on top of Beaver Triple.

    Returns: (hospitals, shared_W, scaler)
    """
    # ── Data loading — real SHHS-1 CSV only ─────────────────────────────────────
    print("[Distributed FL] SHHS 데이터 로드")
    t_data_start = time.perf_counter()
    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    print(f"  [timing] CSV 로드+전처리: {(time.perf_counter()-t_data_start)*1000:.1f} ms")

    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )

    print("[Distributed FL] feature 분배:")
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  hospital {i}: {label}")
    print(f"  DP sigma={dp_sigma}  (embedding gradient에 Gaussian noise, label 추론 방지)")
    print(f"  Top-model: 단일 linear — 병원별 column-slice를 로컬 임베딩에 적용, 통신 없음")

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
    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    y_test  = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)

    # ── Optimizer: all sub-models + shared top linear ─────────────────────────
    params = []
    for h in hospitals:
        params.extend(h.sub.parameters())
    params.extend(shared_W.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    beaver    = BeaverProvider(NUM_CLIENTS)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[Distributed FL] 학습 시작 - {n_epochs} epochs, batch={batch_size}")
    t_train_start = time.perf_counter()
    STEP_NAMES = [
        "local_emb",
        "top_layer", "beaver_sigmoid", "label_ss_loss",
        "backward", "optimizer_step",
    ]
    for epoch in range(1, n_epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        step_ms    = {name: 0.0 for name in STEP_NAMES}
        t_epoch_start = time.perf_counter()

        for start in range(0, n, batch_size):
            idx     = perm[start : start + batch_size]
            y_batch = y_train[idx]
            optimizer.zero_grad()

            # Step 1: each hospital computes its local embedding (private)
            t0 = time.perf_counter()
            local_embs = [
                hospitals[i].local_emb(X_trains[i][idx]) for i in range(NUM_CLIENTS)
            ]

            # DP on gradients: prevents label inference from gradient sign/magnitude
            if dp_sigma > 0:
                for emb in local_embs:
                    emb.register_hook(lambda g: g + torch.randn_like(g) * dp_sigma)
            step_ms["local_emb"] += (time.perf_counter() - t0) * 1000

            # Step 2: each hospital applies its own W_top column-slice to its
            # own local embedding — no cross-hospital exchange needed.
            # sum_j logit_share_j = W @ cat(embs) + b = logit  [by block-linearity]
            t0 = time.perf_counter()
            logit_shares = [
                hospitals[j].logit_share(local_embs[j], j == 0)
                for j in range(NUM_CLIENTS)
            ]
            step_ms["top_layer"] += (time.perf_counter() - t0) * 1000

            # Step 3: Beaver Triple sigmoid — σ(logit) ≈ 0.5 + 0.197x - 0.004x³
            # logit stays in share form; no party reconstructs the raw logit.
            # x³ needs 2 multiplications → 2 triples.
            t0 = time.perf_counter()
            triple1 = beaver.generate_triple(logit_shares[0].shape)
            triple2 = beaver.generate_triple(logit_shares[0].shape)
            pred_shares = beaver.sigmoid_approx(logit_shares, triple1, triple2)
            step_ms["beaver_sigmoid"] += (time.perf_counter() - t0) * 1000

            # Step 4: label is secret-shared too — no party ever holds y alone.
            # MSE gradient is a plain subtraction (pred - y), so summing
            # (pred_share_i - y_share_i) reconstructs exactly pred - y without
            # needing a division protocol the way BCE's gradient would.
            t0 = time.perf_counter()
            y_shares    = additive_split(y_batch.unsqueeze(1), n=NUM_CLIENTS)
            diff_shares = [pred_shares[i] - y_shares[i] for i in range(NUM_CLIENTS)]
            diff        = sum(diff_shares)
            loss        = (diff ** 2).mean()
            step_ms["label_ss_loss"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            loss.backward()
            step_ms["backward"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            optimizer.step()
            step_ms["optimizer_step"] += (time.perf_counter() - t0) * 1000

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_wall_ms = (time.perf_counter() - t_epoch_start) * 1000

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
                f"loss={epoch_loss/n_batches:.4f} | test_acc={acc:.4f} | "
                f"wall={epoch_wall_ms:.1f}ms ({n_batches} batches)"
            )
            breakdown = ", ".join(f"{name}={ms:.1f}ms" for name, ms in step_ms.items())
            print(f"    [timing] step breakdown (sum over epoch): {breakdown}")

    train_total_s = time.perf_counter() - t_train_start
    print(f"[Distributed FL] 학습 완료 — 총 {train_total_s:.1f}s ({n_epochs} epochs)")
    return hospitals, shared_W, scaler
