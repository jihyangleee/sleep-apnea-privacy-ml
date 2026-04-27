import torch
import torch.nn as nn
from model import ServerTopModel


class VerticalFLServer:
    """Vertical FL 서버.
    각 클라이언트의 임베딩을 수집하여 탑모델로 예측·역전파를 수행한다.
    가중치는 항상 평문으로 보관되며 암호화하지 않는다.
    """

    def __init__(self, feature_groups: list, emb_dim: int = 16):
        total_emb = emb_dim * len(feature_groups)
        self.top_model = ServerTopModel(total_emb)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.top_model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_and_loss(self, embeddings: list, y_batch: torch.Tensor):
        """임베딩 연결 → 탑모델 → 손실 계산."""
        concat = torch.cat(embeddings, dim=1)
        logit  = self.top_model(concat)
        return self.criterion(logit, y_batch.to(self.device).unsqueeze(1)), logit

    def evaluate(self, embeddings: list, y: torch.Tensor) -> float:
        """정확도 계산 (그래디언트 없음)."""
        with torch.no_grad():
            concat = torch.cat(embeddings, dim=1)
            preds  = torch.sigmoid(self.top_model(concat))
            return ((preds >= 0.5).squeeze() == y.to(self.device)).float().mean().item()
