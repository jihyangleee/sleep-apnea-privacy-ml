import torch

from dataset import load_shhs_data, partition_data_vertical
from client import VerticalClient
from server import VerticalFLServer
from model import VerticalHeartNet

NUM_CLIENTS = 3
EMB_DIM     = 16

# 기관별 검사 패널 기준 feature 분배 (SHHS_FEATURES 인덱스 기준)
# 공통 feature(age=6, sex=7)는 레이블 보유 기관(client C)이 담당
SLEEP_FEATURE_GROUPS = [
    [0, 1, 2],   # client A — 심박·산소 모니터링: SpO2, resting HR, avg HR
    [3, 4, 5],   # client B — 수면검사실(PSG): total sleep, efficiency, deep ratio
    [6, 7, 8],   # client C — 병원/클리닉: age, sex, BMI + label
]

SLEEP_CLIENT_LABELS = [
    "심박·산소 모니터링(SpO2, resting HR, avg HR)",
    "수면검사실(total sleep, efficiency, deep ratio)",
    "병원(age, sex, BMI) + label",
]


def run_vertical_simulation(csv_path: str, n_epochs: int = 30, batch_size: int = 32) -> VerticalHeartNet:
    """Vertical FL 학습 시뮬레이션.

    - 데이터를 검사 패널 기준으로 3개 기관에 분배
    - 각 기관은 서브모델로 임베딩을 계산
    - 서버가 임베딩을 연결하여 탑모델로 손실을 계산하고 역전파
    - 가중치 암호화 없음 — 입력 데이터 프라이버시는 HE Inference 단계에서 보장
    """
    print("[Vertical FL] 데이터 로드")
    X, y = load_shhs_data(csv_path)
    partitions, _, _, _, _ = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )

    print("[Vertical FL] feature 분배:")
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  client {i}: {label} — {SLEEP_FEATURE_GROUPS[i]}")

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

    print(f"\n[Vertical FL] 학습 시작 — {n_epochs} epochs, batch={batch_size}")
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            y_batch   = y_train_t[batch_idx]

            server.optimizer.zero_grad()
            for c in clients:
                c.zero_grad()

            embeddings = [c.embed(batch_idx) for c in clients]
            loss, _ = server.forward_and_loss(embeddings, y_batch)
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

    # 전체 모델을 VerticalHeartNet으로 통합 (HE Inference에서 사용)
    full_model = VerticalHeartNet(feature_groups=SLEEP_FEATURE_GROUPS, emb_dim=EMB_DIM)
    for i, c in enumerate(clients):
        full_model.sub_models[i].load_state_dict(c.sub_model.state_dict())
    full_model.top_model.load_state_dict(server.top_model.state_dict())

    print("[Vertical FL] 학습 완료")
    return full_model
