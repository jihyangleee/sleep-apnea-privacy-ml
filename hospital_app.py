"""FastAPI server for one hospital node.

Start with:
    HOSPITAL_ID=0 uvicorn hospital_app:app --port 8001
    HOSPITAL_ID=1 uvicorn hospital_app:app --port 8002
    HOSPITAL_ID=2 uvicorn hospital_app:app --port 8003

Requires a trained checkpoint (vertical_model.pt):
    python main.py --mode distributed

Or trigger training via POST /train after starting the servers.
"""

import os                                            # 환경변수 읽기용
import base64                                        # 암호문(ciphertext) base64 인코딩/디코딩용
import threading                                     # 학습 중복 실행 방지용 락
from contextlib import asynccontextmanager           # FastAPI lifespan 컨텍스트 매니저 정의용

import torch                                          # 모델 체크포인트 로드/저장
import torch.nn as nn                                 # 공유 top linear layer 정의
import tenseal as ts                                  # CKKS 동형암호(HE) 연산 라이브러리
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request  # API 서버 프레임워크

from schemas import (
    LogitShareRequest, LogitShareResponse,            # /compute_logit_share 요청/응답 스키마
    TrainResponse,                                     # /train, /reload 응답 스키마
)
from model import HospitalModel                       # 병원별 서브모델(임베딩 + top 가중치 일부) 클래스
from he_client import build_he_context                 # 병원 측 HE 컨텍스트 생성 헬퍼
from simulate import SLEEP_FEATURE_GROUPS, run_distributed_simulation  # 분산 학습 시뮬레이션

# ── Configuration from environment ───────────────────────────────────────────

HOSPITAL_ID = int(os.environ.get("HOSPITAL_ID", "0"))     # 이 서버가 담당하는 병원 번호 (0, 1, 2)
MODEL_PATH  = os.environ.get("MODEL_PATH", "vertical_model.pt")  # 체크포인트 파일 경로
CSV_PATH    = os.environ.get("CSV_PATH") or None            # 학습용 CSV 경로 (없으면 기본 데이터 사용)

# ── State ─────────────────────────────────────────────────────────────────────

hospital:    HospitalModel | None = None              # 로드된 병원 서브모델 (미로드 시 None)
he_ctx:      ts.Context    | None = None               # 병원이 자체 생성한 HE 컨텍스트
client_ctx:  ts.Context    | None = None  # patient's uploaded public context  # 환자가 업로드한 공개 컨텍스트 (있으면 우선 사용)
_train_lock = threading.Lock()                         # 동시에 두 번 학습이 실행되지 않도록 막는 락


# ── Model loading ─────────────────────────────────────────────────────────────
#
def _load_model():
    global hospital, he_ctx                            # 모듈 전역 상태(hospital, he_ctx)를 갱신

    ckpt      = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)  # 체크포인트 파일 로드
    fg        = ckpt["feature_groups"]                  # 병원별 feature 그룹 정보
    emb_dim   = ckpt["emb_dim"]                          # 서브모델 임베딩 차원
    n         = len(fg)                                   # 병원(feature 그룹) 개수
    total_emb = emb_dim * n                                # top layer 입력 전체 차원 (병원 수 × emb_dim)

    shared_W = nn.Linear(total_emb, 1)                    # 모든 병원이 공유하는 top linear layer 정의
    shared_W.load_state_dict(ckpt["top_W"])                # 학습된 top layer 가중치 로드

    h = HospitalModel(HOSPITAL_ID, fg, emb_dim, shared_W)  # 이 병원이 담당할 서브모델 생성
    h.sub.load_state_dict(ckpt[f"sub_{HOSPITAL_ID}"])        # 이 병원의 서브모델 가중치만 로드
    h.eval()                                                 # 추론 모드로 전환 (dropout/batchnorm 등 비활성화)

    # Each hospital holds its own column slice of the top linear weight
    W_top   = shared_W.weight.detach().numpy()   # (1, total_emb)  top layer 전체 가중치
    W_top_i = W_top[:, HOSPITAL_ID * emb_dim : (HOSPITAL_ID + 1) * emb_dim]  # (1, emb_dim)  이 병원 몫만 슬라이싱
    b_top   = shared_W.bias.detach().numpy()     # (1,)  top layer bias (병원 0만 실제로 사용해서 더함)

    h.build_he_weights(W_top_i, b_top)                      # 슬라이싱한 가중치를 HE 연산 가능한 형태로 준비

    hospital = h                                             # 전역 상태에 서브모델 등록
    he_ctx   = build_he_context()                             # 이 병원 자체 HE 컨텍스트 생성
    print(f"[Hospital {HOSPITAL_ID}] Model loaded from {MODEL_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):                         # 체크포인트 파일이 이미 존재하면
        _load_model()                                        # 서버 기동 시점에 바로 모델 로드
    else:
        print(f"[Hospital {HOSPITAL_ID}] No checkpoint at {MODEL_PATH}. POST /train to train first.")
    yield                                                     # 이 지점 이후 서버가 요청을 처리(앱 실행 구간)


def _require_model():
    if hospital is None:                                     # 모델이 아직 로드되지 않았다면
        raise HTTPException(503, "Model not loaded. POST /train first.")  # 503으로 요청 거부


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title=f"Hospital {HOSPITAL_ID}", lifespan=lifespan)  # 병원별 FastAPI 앱 인스턴스 생성


# ── Inter-hospital endpoint ───────────────────────────────────────────────────

def _get_ctx() -> ts.Context:
    if client_ctx is not None:                               # 환자가 자신의 공개 HE 컨텍스트를 업로드했다면
        return client_ctx                                      # 그 컨텍스트를 우선 사용 (환자가 복호화 가능해야 하므로)
    return he_ctx                                             # 없으면 병원 자체 컨텍스트 사용


@app.post("/upload_context")
async def upload_context(request: Request):
    """Patient uploads serialized public CKKS context as raw bytes."""
    global client_ctx                                        # 전역 client_ctx 갱신
    data = await request.body()                               # 요청 바디를 raw bytes로 읽음
    client_ctx = ts.context_from(data)                        # 직렬화된 CKKS 컨텍스트를 역직렬화하여 저장
    print(f"[Hospital {HOSPITAL_ID}] Patient context uploaded ({len(data)//1024} KB).")
    return {"status": "ok"}


@app.post("/compute_logit_share", response_model=LogitShareResponse)
async def compute_logit_share(req: LogitShareRequest):
    """enc(features_i) → enc(emb_i) → enc(logit_i) in one shot.

    Patient (Watch) calls this directly on each hospital with that hospital's
    own encrypted feature slice. No coordinator or inter-hospital communication.

    Hospital 0 adds b_top to its ciphertext so the patient can simply sum all
    three responses to get enc(logit_final), then decrypt with their secret key.
    """
    _require_model()                                          # 모델이 로드되어 있는지 확인
    ctx    = _get_ctx()                                        # 사용할 HE 컨텍스트 결정
    enc_xi = base64.b64decode(req.enc_xi_b64 + "=" * (-len(req.enc_xi_b64) % 4))  # base64 → 암호문 바이트 (패딩 보정 포함)
    enc_emb = hospital.compute_sub_emb_he(enc_xi, ctx)         # 암호화된 feature → 암호화된 임베딩
    enc_logit_share = hospital.compute_logit_share_he(enc_emb, ctx)  # 암호화된 임베딩 → 암호화된 logit 부분값

    # Hospital 0 absorbs b_top into its ciphertext so the patient just sums
    if HOSPITAL_ID == 0 and hospital._b_top is not None:      # 병원 0만 bias를 담당
        enc_l = ts.lazy_ckks_vector_from(enc_logit_share)       # 직렬화된 결과를 다시 CKKS 벡터로 복원
        enc_l.link_context(ctx)                                  # 연산을 위해 컨텍스트 연결
        enc_l += hospital._b_top                                 # bias를 암호문 상태에서 더함
        enc_logit_share = enc_l.serialize()                       # 다시 직렬화

    return LogitShareResponse(
        enc_logit_share_b64=base64.b64encode(enc_logit_share).decode()  # 암호문을 base64 문자열로 응답
    )


# ── Training endpoint ─────────────────────────────────────────────────────────

def _run_training(csv_path=None, dreamt_dir=None, dp_sigma=0.01, n_epochs=30):
    with _train_lock:                                          # 동시 학습 방지
        hospitals_tr, W, scaler = run_distributed_simulation(   # 세 병원의 분산(수직) 학습 시뮬레이션 실행
            csv_path=csv_path, dreamt_dir=dreamt_dir,
            n_epochs=n_epochs, dp_sigma=dp_sigma,
        )
        torch.save(
            {
                "mode":           "distributed",                # 체크포인트 모드 표시
                "feature_groups": SLEEP_FEATURE_GROUPS,          # feature 그룹 메타데이터 저장
                "emb_dim":        hospitals_tr[0].emb_dim,       # 임베딩 차원 저장
                "sub_0":          hospitals_tr[0].sub.state_dict(),  # 병원 0 서브모델 가중치
                "sub_1":          hospitals_tr[1].sub.state_dict(),  # 병원 1 서브모델 가중치
                "sub_2":          hospitals_tr[2].sub.state_dict(),  # 병원 2 서브모델 가중치
                "top_W":          W.state_dict(),                 # 공유 top layer 가중치
                "scaler_mean":    scaler.mean_.tolist(),          # feature 정규화 평균값
                "scaler_scale":   scaler.scale_.tolist(),         # feature 정규화 스케일값
            },
            MODEL_PATH,                                          # 체크포인트 저장 경로
        )
        _load_model()                                            # 새로 저장한 체크포인트를 즉시 재로드
        print(f"[Hospital {HOSPITAL_ID}] Retraining complete.")


@app.post("/train", response_model=TrainResponse)
async def train(background_tasks: BackgroundTasks):
    """Trigger distributed FL training. Saves checkpoint and reloads model."""
    if _train_lock.locked():                                    # 이미 학습이 진행 중이면
        return TrainResponse(status="running", message="Training already in progress.")  # 중복 실행 없이 바로 응답
    background_tasks.add_task(_run_training, csv_path=CSV_PATH)  # 학습을 백그라운드 태스크로 등록 (응답은 즉시 반환)
    return TrainResponse(status="started", message="Training started in background. GET /health to check.")


@app.post("/reload", response_model=TrainResponse)
async def reload_model():
    """Reload model from checkpoint (useful after another hospital triggers training)."""
    if not os.path.exists(MODEL_PATH):                          # 체크포인트 파일이 없으면
        raise HTTPException(404, f"No checkpoint at {MODEL_PATH}")  # 404 반환
    _load_model()                                                # 디스크의 체크포인트로부터 모델 재로드
    return TrainResponse(status="done", message=f"Model reloaded from {MODEL_PATH}")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "hospital_id":  HOSPITAL_ID,                            # 이 서버의 병원 번호
        "model_loaded": hospital is not None,                    # 모델 로드 여부
        "model_path":   MODEL_PATH,                              # 체크포인트 경로
    }
