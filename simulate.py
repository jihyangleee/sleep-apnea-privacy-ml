import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
    load_shhs_data, load_dreamt_data, generate_dummy_data,
    partition_data_vertical, generate_galaxy_watch_users,
)
from client import VerticalClient
from server import VerticalFLServer
from model import VerticalHeartNet, HospitalModel
from secret_sharing import additive_split, apply_dp_noise, BeaverProvider

NUM_CLIENTS = 3
EMB_DIM     = 16

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


def run_vertical_simulation(
    csv_path: str = None,
    dreamt_dir: str = None,
    n_epochs: int = 30,
    batch_size: int = 32,
):
    """Vertical FL training with additive Secret Sharing.

    Privacy model:
      - Each client only sees its own feature columns.
      - Embeddings are additively secret-shared before being "sent" to
        the coordinator; the coordinator never sees a full embedding from
        any single client.
      - Top-model linear1 is evaluated in shares (partial_i = W1 @ share_i);
        aggregation at the coordinator reconstructs the full hidden vector
        before the non-linear PolyAct, which requires a trusted coordinator
        (or a separate MPC multiplication protocol in production).

    Returns: (VerticalHeartNet, scaler) — model for HE inference, scaler for
             normalising new individual inputs.
    """
    # ── Data loading ─────────────────────────────────────────────────────────
    if dreamt_dir is not None:
        print("[Vertical FL] DREAMT 데이터 로드")
        X, y, scaler = load_dreamt_data(dreamt_dir, return_scaler=True)
    elif csv_path is not None:
        print("[Vertical FL] SHHS 데이터 로드")
        X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    else:
        print("[Vertical FL] CSV not provided - using synthetic dummy data")
        X, y, scaler = generate_dummy_data(return_scaler=True)

    partitions, _, _, _, _ = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )

    print("[Vertical FL] feature 분배:")
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  client {i}: {label} - {SLEEP_FEATURE_GROUPS[i]}")

    # ── Build clients & server ───────────────────────────────────────────────
    clients = [
        VerticalClient(
            client_id=i,
            X_train=p["X_train"],
            X_test=p["X_test"],
            feature_indices=p["feature_indices"],
            emb_dim=EMB_DIM,
        )
        for i, p in enumerate(partitions)
    ]
    server = VerticalFLServer(feature_groups=SLEEP_FEATURE_GROUPS, emb_dim=EMB_DIM)

    y_train_t = torch.tensor(partitions[0]["y_train"], dtype=torch.float32)
    y_test_t  = torch.tensor(partitions[0]["y_test"],  dtype=torch.float32)
    n = len(y_train_t)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[Vertical FL] 학습 시작 - {n_epochs} epochs, batch={batch_size}")
    print(  "              3-way additive SS per embedding")
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            y_batch   = y_train_t[batch_idx]

            server.optimizer.zero_grad()
            for c in clients:
                c.zero_grad()

            # ── Step 1: Each client computes its embedding (grad tracked) ──
            embeddings = [c.embed(batch_idx) for c in clients]

            # ── Step 2: Additive SS — each share already random, wire-safe ──
            # all_shares[i][j] = share of embedding i going to client j
            all_shares = [additive_split(emb, n=NUM_CLIENTS) for emb in embeddings]

            # ── Step 3: Collect shares; concat per receiving client ─────────
            concat_shares = []
            for j in range(NUM_CLIENTS):
                received = [all_shares[i][j] for i in range(NUM_CLIENTS)]
                concat_shares.append(torch.cat(received, dim=1))
                # concat_shares[j]: (batch, NUM_CLIENTS * EMB_DIM)

            # ── Step 4: Distributed linear — each party applies top W to its share ──
            # sum_j( W @ share_j ) = W @ cat(embs)  [linearity; bias added once]
            partial_logits = [
                F.linear(cs, server.top_model.linear.weight) for cs in concat_shares
            ]
            partial_logits[0] = partial_logits[0] + server.top_model.linear.bias

            # ── Step 5: Coordinator aggregates → logit ────────────────────────
            logit = sum(partial_logits)   # → (batch, 1)

            # ── Step 6: Loss + backward ────────────────────────────────────
            loss = server.criterion(logit, y_batch.to(server.device).unsqueeze(1))
            loss.backward()

            server.optimizer.step()
            for c in clients:
                c.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % 5 == 0 or epoch == 1:
            test_embs = [c.embed_test().detach() for c in clients]
            acc = server.evaluate(test_embs, y_test_t)
            print(f"  Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss/n_batches:.4f} | test_acc={acc:.4f}")

    # ── Consolidate weights into VerticalHeartNet (used by HE inference) ────
    full_model = VerticalHeartNet(feature_groups=SLEEP_FEATURE_GROUPS, emb_dim=EMB_DIM)
    for i, c in enumerate(clients):
        full_model.sub_models[i].load_state_dict(c.sub_model.state_dict())
    full_model.top_model.load_state_dict(server.top_model.state_dict())

    print("[Vertical FL] 학습 완료")
    return full_model, scaler


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
