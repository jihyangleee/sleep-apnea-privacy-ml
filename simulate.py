import random
import torch
import torch.nn as nn

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

            # ── Step 4: Distributed linear1 — each client applies W1 to its share ──
            # sum_j linear1(concat_shares[j]) == linear1(cat(embeddings))  [linearity]
            # This is the core SS property: linear computation decomposes over shares.
            partial_h = [server.top_model.linear1(cs) for cs in concat_shares]

            # ── Step 5: Coordinator aggregates, applies PolyAct, then Linear2 ──
            h_linear = sum(partial_h)
            h_act    = server.top_model.act(h_linear)
            logit    = server.top_model.linear2(h_act)   # → (batch, 1)

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
    top_hidden: int = 32,
):
    """Fully distributed VFL training — no central server sees any embedding.

    Privacy model
    -------------
    - Sub-model runs locally on each hospital's own features (private).
    - Embeddings are additively secret-shared before transmission.
    - DP noise (sigma) is injected into each share in transit, so even a
      curious coordinator only sees noisy shares — not the real embedding.
    - Top-model non-linear (PolyAct) is computed via Beaver Triple: parties
      exchange O(hidden_dim) values instead of reconstructing the full embedding.

    Returns: (hospitals, shared_top_W1, shared_top_W2, scaler)
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
    print(f"  Top-model: 완전 linear — SS만으로 분산 계산, Beaver Triple 불필요")

    # ── Build hospitals with SHARED top-model weights ─────────────────────────
    total_emb = emb_dim * NUM_CLIENTS
    shared_W1 = nn.Linear(total_emb, top_hidden)
    shared_W2 = nn.Linear(top_hidden, 1)

    hospitals = [
        HospitalModel(
            i, SLEEP_FEATURE_GROUPS, emb_dim, top_hidden, shared_W1, shared_W2
        )
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

    # ── Optimizer: all sub-models + shared top model ──────────────────────────
    params = []
    for h in hospitals:
        params.extend(h.sub.parameters())
    params.extend(shared_W1.parameters())
    params.extend(shared_W2.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
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

            # Gradient DP: add noise to ∂L/∂emb_i before it flows into bottom models.
            # Prevents passive hospitals from inferring the label via gradient magnitude/sign.
            if dp_sigma > 0:
                for emb in local_embs:
                    emb.register_hook(lambda g: g + torch.randn_like(g) * dp_sigma)

            # Step 2: additive SS split + DP noise before transmission
            all_shares = [additive_split(emb, n=NUM_CLIENTS) for emb in local_embs]

            # Step 3: DP noise -> receive and concatenate
            concat_shares = []
            for j in range(NUM_CLIENTS):
                received = [
                    apply_dp_noise(all_shares[i][j], dp_sigma)
                    for i in range(NUM_CLIENTS)
                ]
                concat_shares.append(torch.cat(received, dim=1))
            # concat_shares[j]: (batch, NUM_CLIENTS * emb_dim) + DP noise

            # Step 4: each hospital applies Linear1 to its share (no bias except j=0)
            h_shares = [
                hospitals[j].h_share(concat_shares[j], j == 0)
                for j in range(NUM_CLIENTS)
            ]

            # Step 5: Beaver Triple — PolyAct without any party seeing h_linear
            # h_linear은 재구성되지 않음; 각 party는 act_share만 보유
            triples    = beaver.generate_triple(h_shares[0].shape)
            act_shares = beaver.poly_act(h_shares, triples)

            # Step 6: each hospital applies Linear2 to its act_share
            logit_shares = [
                hospitals[j].logit_share(act_shares[j], j == 0)
                for j in range(NUM_CLIENTS)
            ]

            # Step 7: randomly chosen coordinator sums logit shares → scalar logit
            # coordinator가 보는 건 h_linear(32d) 아닌 logit(스칼라)뿐
            coord = random.randrange(NUM_CLIENTS)
            logit = sum(logit_shares)
            loss  = criterion(logit, y_batch.unsqueeze(1))
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
                h1     = shared_W1(test_embs)
                h1_act = h1 * (h1 + 0.5)
                logit  = shared_W2(h1_act)
                pred   = (torch.sigmoid(logit.squeeze()) > 0.5).float()
                acc    = (pred == y_test).float().mean().item()
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"loss={epoch_loss/n_batches:.4f} | test_acc={acc:.4f}"
            )

    print("[Distributed FL] 학습 완료")
    return hospitals, shared_W1, shared_W2, scaler
