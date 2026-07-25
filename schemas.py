from pydantic import BaseModel


class LogitShareRequest(BaseModel):
    enc_xi_b64: str               # base64: this hospital's enc(features_i)

class LogitShareResponse(BaseModel):
    enc_logit_share_b64: str      # base64: enc(logit_i) from this hospital


class InferRequest(BaseModel):
    enc_xi_b64: dict[str, str]    # hospital_id (str) -> base64 enc(features_i), for ALL hospitals

class InferTiming(BaseModel):
    local_compute_ms: float       # entry-point 병원 자신의 HE 계산 시간
    peer_calls_ms: dict[str, float]  # peer hospital_id (str) -> 그 병원까지의 HTTP 왕복 시간
    total_ms: float               # /infer 요청 전체 처리 시간 (entry-point 서버 기준)

class InferResponse(BaseModel):
    enc_logit_shares_b64: dict[str, str]  # hospital_id (str) -> base64 enc(logit_share) — 합산은 client가 함
    timing: InferTiming                   # 병원 간 통신 오버헤드 breakdown


class TrainResponse(BaseModel):
    status: str                   # "started" | "running" | "done"
    message: str


class TrainInitRequest(BaseModel):
    emb_dim: int = 16

class TrainForwardRequest(BaseModel):
    batch_token: str              # coordinator가 발급, forward/backward 짝을 맞추기 위한 키
    indices: list[int]            # 이 배치에 해당하는 train-row 인덱스 (coordinator와 동일 split 전제)

class TrainForwardResponse(BaseModel):
    emb: list[list[float]]        # 이 병원의 로컬 forward 결과 (평문 — 원본 feature는 이 병원 밖으로 안 나감)

class TrainBackwardRequest(BaseModel):
    batch_token: str
    grad: list[list[float]]       # coordinator가 계산한, 이 병원 임베딩에 대한 gradient
