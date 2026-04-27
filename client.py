import torch
from model import ClientSubModel


class VerticalClient:
    """Vertical FL 클라이언트.
    담당 feature 열만 보유하고, 서브모델로 임베딩을 계산한다.
    실제 네트워크 통신 없이 simulate.py가 직접 호출하는 시뮬레이션용.
    """

    def __init__(self, client_id: int, X_train, X_test, feature_indices: list, emb_dim: int = 16):
        self.client_id = client_id
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test  = torch.tensor(X_test,  dtype=torch.float32)
        self.feature_indices = feature_indices
        self.sub_model = ClientSubModel(len(feature_indices), emb_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sub_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.sub_model.parameters(), lr=0.001)

    def embed(self, batch_idx: torch.Tensor) -> torch.Tensor:
        """학습 배치 임베딩 계산 (그래디언트 추적 유지)."""
        x = self.X_train[batch_idx].to(self.device)
        return self.sub_model(x)

    def embed_test(self) -> torch.Tensor:
        """테스트 전체 임베딩 계산."""
        return self.sub_model(self.X_test.to(self.device))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
