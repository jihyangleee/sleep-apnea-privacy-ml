import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    """HE 추론 호환 활성화 함수 (x² — 다항식이므로 CKKS로 그대로 계산 가능)."""
    def forward(self, x):
        return x * x


class ClientSubModel(nn.Module):
    """Vertical FL 클라이언트 서브모델: 담당 feature 열 → 임베딩."""
    def __init__(self, input_dim: int, emb_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        return self.linear(x) ** 2


class ServerTopModel(nn.Module):
    """Vertical FL 서버 탑모델: 연결된 임베딩 → 로짓 (Sigmoid 없음, HE 호환)."""
    def __init__(self, total_emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(total_emb_dim, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.linear2(self.linear1(x) ** 2)


class VerticalHeartNet(nn.Module):
    """Vertical FL 전체 모델 — 서브모델 + 탑모델 통합 (추론·저장·로드용)."""
    def __init__(self, feature_groups: list, emb_dim: int = 16):
        super().__init__()
        self.feature_groups = [list(fg) for fg in feature_groups]
        self.emb_dim = emb_dim
        self.sub_models = nn.ModuleList([
            ClientSubModel(len(fg), emb_dim) for fg in self.feature_groups
        ])
        self.top_model = ServerTopModel(emb_dim * len(self.feature_groups))

    def forward(self, x):
        embs = [sub(x[:, fg]) for sub, fg in zip(self.sub_models, self.feature_groups)]
        return self.top_model(torch.cat(embs, dim=1))
