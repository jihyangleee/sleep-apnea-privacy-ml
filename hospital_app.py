"""FastAPI server for one hospital node.

Start with:
    HOSPITAL_ID=0 uvicorn hospital_app:app --port 8001
    HOSPITAL_ID=1 uvicorn hospital_app:app --port 8002
    HOSPITAL_ID=2 uvicorn hospital_app:app --port 8003

Requires a trained checkpoint (vertical_model.pt):
    python main.py --mode distributed

Or trigger training via POST /train after starting the servers.
"""

import io                                            # 서브모델 state_dict를 바이트로 주고받기 위함 (/train_export)
import os
import uuid                                          # 학습 배치별 forward/backward 짝을 맞추는 토큰
import time
import base64
import threading
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import tenseal as ts                                  # CKKS 동형암호 라이브러리
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response

from schemas import (
    LogitShareRequest, LogitShareResponse,
    InferRequest, InferResponse, InferTiming,
    TrainResponse,
    TrainInitRequest, TrainForwardRequest, TrainForwardResponse, TrainBackwardRequest,
)

from model import HospitalModel, ClientSubModel
from he_client import build_he_context
from dataset import load_shhs_data, partition_data_vertical
from secret_sharing import additive_split, apply_dp_noise, BeaverProvider
from simulate import SLEEP_FEATURE_GROUPS, run_distributed_simulation

# ── Configuration from environment ───────────────────────────────────────────

HOSPITAL_ID = int(os.environ.get("HOSPITAL_ID", "0"))     # 이 서버가 담당하는 병원 번호 (0, 1, 2)
MODEL_PATH  = os.environ.get("MODEL_PATH", "vertical_model.pt")  # 체크포인트 파일 경로
CSV_PATH    = os.environ.get("CSV_PATH") or None            # csv 경로

# hospital_1 , hospital_2, hospial_3 가 기본값
HOSPITAL_HOSTS = os.environ.get(
    "HOSPITAL_HOSTS", "hospital_0,hospital_1,hospital_2"
).split(",")


def _peer_url(peer_id: int) -> str:
    return f"http://{HOSPITAL_HOSTS[peer_id]}:8000"


def _b64_decode_padded(s: str) -> bytes:
    return base64.b64decode(s + "=" * (-len(s) % 4))

# ── State ─────────────────────────────────────────────────────────────────────

hospital:    HospitalModel | None = None              # 로드된 병원 서브모델
he_ctx:      ts.Context    | None = None               # 병원이 자체 생성한 HE 컨텍스트 (실제로는 사용되지 않음)
client_ctx:  ts.Context    | None = None               # 환자가 업로드한 공개 컨텍스트 (암호화 스킴 종류, 키값, 등 - 설정묶음)
# 덧셈 / 곱셈을 하려면 양쪽이 정확히 같은 context를 가져야 함 
_train_lock = threading.Lock()

# HTTP 분산 학습 시 이 병원이 "peer"(진입점 아님)로서 갖는 상태 —
# 추론용 hospital 객체와는 별개로, 학습 전용 서브모델을 따로 둔다.
_train_sub:       ClientSubModel        | None = None   # 이 병원의 학습 중인 서브모델
_train_optimizer: torch.optim.Optimizer | None = None   # 그 서브모델만을 위한 로컬 optimizer
_train_X:         torch.Tensor          | None = None   # 이 병원의 train feature 슬라이스 (원본 밖으로 안 나감)
_train_cache:     dict[str, torch.Tensor] = {}           # batch_token -> forward 출력 (backward에서 재사용)
 

# ── Model loading ─────────────────────────────────────────────────────────────
# 병원 추론 모델 
def _load_model():
    global hospital, he_ctx                           

    ckpt      = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)   # 이미 학습 완료된 결과물을 가져옴
    fg        = ckpt["feature_groups"]                  # 병원별 feature 그룹 정보
    emb_dim   = ckpt["emb_dim"]                          # 서브모델 임베딩 차원
    n         = len(fg)                                   # 병원(feature 그룹) 개수
    total_emb = emb_dim * n                                # top layer 입력 전체 차원 (병원 수 × emb_dim)

    shared_W = nn.Linear(total_emb, 1)                    # 모든 병원이 공유하는 top linear layer 정의
    shared_W.load_state_dict(ckpt["top_W"])                # 학습된 top layer 가중치 로드

    h = HospitalModel(HOSPITAL_ID, fg, emb_dim, shared_W)  # 이 병원이 담당할 서브모델 생성
    h.sub.load_state_dict(ckpt[f"sub_{HOSPITAL_ID}"])        # 이 병원의 서브모델 가중치만 로드
    h.eval()                                                 # 추론 모드로 전환 (dropout/batchnorm 등 비활성화)

    
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


# client 의 context upload 코드 
@app.post("/upload_context") # client 측면 raw bytes 
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
    enc_xi = _b64_decode_padded(req.enc_xi_b64)                # base64 → 암호문 바이트 (패딩 보정 포함)
    enc_emb = hospital.compute_sub_emb_he(enc_xi, ctx)         # 암호화된 feature → 암호화된 임베딩
    # 서브 모델에서 enx_xi - feature 값을 ctx 정보를 이용해 연산함 (비선형- 근사 다항식)
    # 밑 코드에서 병원들이 서로 임베딩 값들을 주고받지 않고 각자의 임베딩 값을 top model에 넣는다.
    enc_logit_share = hospital.compute_logit_share_he(enc_emb, ctx)  # 암호화된 임베딩 → 암호화된 logit 부분값

    
    # y 절편값 
    if HOSPITAL_ID == 0 and hospital._b_top is not None:      # 병원 0만 bias를 담당 # 이미 학습할 때 저장됨 (b_top - 절편)
        enc_l = ts.lazy_ckks_vector_from(enc_logit_share)       # 직렬화된 결과를 다시 CKKS 벡터로 복원
        enc_l.link_context(ctx)                                  # 연산을 위해 컨텍스트 연결
        enc_l += hospital._b_top                                 # bias를 암호문 상태에서 더함
        enc_logit_share = enc_l.serialize()                       # 다시 직렬화

    return LogitShareResponse(
        enc_logit_share_b64=base64.b64encode(enc_logit_share).decode()  # 암호문을 base64 문자열로 응답
    )


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """진입점(entry-point) 엔드포인트 — 환자가 세 병원 몫을 전부 여기 한 번에 보냄.

    이 병원은 자기 몫을 로컬로 계산하고, 나머지 두 병원에는 기존
    /compute_logit_share를 HTTP로 호출해서 각자의 enc(logit_share)를 받아온다.
    합산은 여기서 하지 않고 세 개의 enc(logit_share)를 그대로 client에게
    돌려준다 — entry-point도 비밀키가 없어 어차피 복호화를 못 하니, 굳이
    entry-point가 암호문 덧셈을 대신 해줄 이유가 없다. direct 패턴(환자가
    세 병원에 각각 요청)과 동일하게 최종 합산+복호화는 client 몫이다.

    환자는 /infer를 호출하기 전에 세 병원 모두에게 /upload_context로 자신의
    공개 CKKS 컨텍스트를 먼저 업로드해둬야 한다 (그래야 세 병원이 같은
    컨텍스트로 암호문을 만들어서 client가 서로 더할 수 있음).

    응답의 timing 필드에 로컬 HE 연산 시간과 병원별 HTTP 왕복 시간을 담아서
    반환한다 — 병원 간 통신 오버헤드를 측정하기 위함.
    """
    _require_model()
    ctx = _get_ctx()
    t_start = time.perf_counter()
    # 각 병원은 enc(feature) 값을 전달 받음 enc_xi_b64  
    my_xi_b64 = req.enc_xi_b64.get(str(HOSPITAL_ID))          # 이 병원 몫이 요청에 포함돼 있는지 확인
    if my_xi_b64 is None:
        raise HTTPException(400, f"enc_xi_b64 missing for this hospital ({HOSPITAL_ID}).")

    t_local_start = time.perf_counter()
    enc_xi  = _b64_decode_padded(my_xi_b64)
    enc_emb = hospital.compute_sub_emb_he(enc_xi, ctx)  # sub model
    my_share = hospital.compute_logit_share_he(enc_emb, ctx) # 
    if HOSPITAL_ID == 0 and hospital._b_top is not None:       # 병원 0이면 자기 몫에 bias까지 로컬로 처리
        enc_l = ts.lazy_ckks_vector_from(my_share)
        enc_l.link_context(ctx)
        enc_l += hospital._b_top
        my_share = enc_l.serialize()
    local_compute_ms = (time.perf_counter() - t_local_start) * 1000

    enc_logit_shares_b64: dict[str, str] = {                   # 합산 없이 각자의 share를 그대로 모음
        str(HOSPITAL_ID): base64.b64encode(my_share).decode()
    }

    peer_calls_ms: dict[str, float] = {}
    n_hospitals = len(SLEEP_FEATURE_GROUPS)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for peer_id in range(n_hospitals):
            if peer_id == HOSPITAL_ID:
                continue
            peer_xi_b64 = req.enc_xi_b64.get(str(peer_id))
            if peer_xi_b64 is None:
                raise HTTPException(400, f"enc_xi_b64 missing for hospital {peer_id}.")

            t_peer_start = time.perf_counter()                  # 병원 간(entry → peer) 통신 오버헤드 측정 시작
            resp = await client.post(                          # 다른 병원의 /compute_logit_share를 그대로 재사용
                f"{_peer_url(peer_id)}/compute_logit_share",
                json={"enc_xi_b64": peer_xi_b64},
            )
            resp.raise_for_status()
            peer_calls_ms[str(peer_id)] = (time.perf_counter() - t_peer_start) * 1000
            # 실제 client 한테 가는 값 
            enc_logit_shares_b64[str(peer_id)] = resp.json()["enc_logit_share_b64"]  # 릴레이만, 합산 안 함

    total_ms = (time.perf_counter() - t_start) * 1000
    print(                                                      # 컨테이너 로그(docker compose logs)에서 오버헤드 확인용
        f"[Hospital {HOSPITAL_ID}] /infer timing — "
        f"local={local_compute_ms:.1f}ms, "
        f"peers={ {k: round(v, 1) for k, v in peer_calls_ms.items()} }, "
        f"total={total_ms:.1f}ms"
    )

    return InferResponse(
        enc_logit_shares_b64=enc_logit_shares_b64,
        timing=InferTiming(
            local_compute_ms=local_compute_ms,
            peer_calls_ms=peer_calls_ms,
            total_ms=total_ms,
        ),
    )


# ── Training endpoint ─────────────────────────────────────────────────────────

def _run_training(csv_path, dp_sigma=0.01, n_epochs=30):
    with _train_lock:                                          # 동시 학습 방지
        hospitals_tr, W, scaler = run_distributed_simulation(   # 세 병원의 분산(수직) 학습 시뮬레이션 실행
            csv_path=csv_path,
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
    if CSV_PATH is None:                                        # 실제 SHHS CSV 경로가 설정 안 됐으면
        raise HTTPException(400, "CSV_PATH env var not set. Real SHHS CSV is required for training.")
    if _train_lock.locked():                                    # 이미 학습이 진행 중이면
        return TrainResponse(status="running", message="Training already in progress.")  # 중복 실행 없이 바로 응답
    background_tasks.add_task(_run_training, csv_path=CSV_PATH)  # 학습을 백그라운드 태스크로 등록 (응답은 즉시 반환)
    return TrainResponse(status="started", message="Training started in background. GET /health to check.")


# ── HTTP-distributed training (peer side) ────────────────────────────────────
# /train 은 세 병원을 전부 한 프로세스 안에서 시뮬레이션하지만, 아래 4개
# 엔드포인트는 이 병원이 "peer"로서 자기 서브모델의 forward/backward를
# 실제로 로컬에서 수행하고, 그 결과(임베딩/gradient)만 entry-point 병원과
# HTTP로 주고받기 위한 것이다. 원본 feature와 라벨은 이 병원 밖으로 나가지 않는다.

@app.post("/train_init", response_model=TrainResponse)
async def train_init(req: TrainInitRequest):
    """entry-point 병원이 학습 시작 전에 각 peer에게 호출 — 이 병원 몫의
    데이터를 로드하고, 이번 학습 세션 전용 서브모델+optimizer를 새로 만든다."""
    global _train_sub, _train_optimizer, _train_X
    if CSV_PATH is None:
        raise HTTPException(400, "CSV_PATH env var not set. Real SHHS CSV is required for training.")

    X, y, _ = load_shhs_data(CSV_PATH, return_scaler=True)
    partitions, _, _, _, _ = partition_data_vertical(
        X, y, num_clients=len(SLEEP_FEATURE_GROUPS), feature_groups=SLEEP_FEATURE_GROUPS
    )
    _train_X = torch.tensor(partitions[HOSPITAL_ID]["X_train"], dtype=torch.float32)

    input_dim     = len(SLEEP_FEATURE_GROUPS[HOSPITAL_ID])
    _train_sub    = ClientSubModel(input_dim, req.emb_dim)
    _train_optimizer = torch.optim.Adam(_train_sub.parameters(), lr=1e-3)
    _train_cache.clear()

    return TrainResponse(
        status="done",
        message=f"train_init complete — {_train_X.shape[0]} train rows loaded locally for hospital {HOSPITAL_ID}",
    )


@app.post("/train_forward", response_model=TrainForwardResponse)
async def train_forward(req: TrainForwardRequest):
    """이 병원 몫의 배치에 대해 로컬 forward만 수행하고 임베딩을 평문으로 반환.
    원본 feature(X)는 절대 이 함수 밖으로 안 나가고, 나가는 건 임베딩뿐이다.
    반환한 텐서는 batch_token으로 캐싱해뒀다가 /train_backward에서 그 위에
    직접 .backward(grad)를 호출해 로컬 optimizer를 갱신하는 데 쓴다."""
    if _train_sub is None:
        raise HTTPException(503, "Call /train_init first.")

    idx = torch.tensor(req.indices, dtype=torch.long)
    x_batch = _train_X[idx]
    emb = _train_sub(x_batch)                # 로컬 forward (autograd graph는 이 프로세스 안에만 존재)
    _train_cache[req.batch_token] = emb
    return TrainForwardResponse(emb=emb.detach().tolist())


@app.post("/train_backward", response_model=TrainResponse)
async def train_backward(req: TrainBackwardRequest):
    """entry-point 병원이 계산해서 보내준 gradient를, 캐싱해둔 로컬 forward
    텐서에 그대로 흘려서 이 병원의 서브모델 파라미터만 갱신한다."""
    emb = _train_cache.pop(req.batch_token, None)
    if emb is None:
        raise HTTPException(400, f"Unknown batch_token {req.batch_token} (not found or already consumed).")

    grad = torch.tensor(req.grad, dtype=torch.float32)
    _train_optimizer.zero_grad()
    emb.backward(grad)                        # 이 병원 로컬 그래프에서만 역전파, 다른 병원은 관여 안 함
    _train_optimizer.step()
    return TrainResponse(status="done", message="backward applied locally")


@app.get("/train_export")
async def train_export():
    """학습이 끝난 뒤 entry-point 병원이 이 병원의 최종 서브모델 가중치를
    가져가서 하나의 체크포인트로 조립할 때 씀."""
    if _train_sub is None:
        raise HTTPException(503, "No trained sub-model yet. Call /train_init and complete training first.")
    buf = io.BytesIO()
    torch.save(_train_sub.state_dict(), buf)
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


# ── HTTP-distributed training (entry-point side) ──────────────────────────────
# /infer와 같은 패턴: 이 함수를 실행하는 병원은 고정된 역할이 아니라
# /train_distributed를 받은 그 순간에만 진입점이 된다. 자기 몫은 로컬로
# 처리하고, 나머지 두 병원과는 forward/backward를 실제 HTTP로 주고받는다.

async def _run_training_http_distributed(csv_path, dp_sigma=0.01, n_epochs=30, batch_size=32, emb_dim=16):
    """/train과 동일한 수학(SS + Beaver Triple + label SS + MSE)을 쓰지만,
    peer 두 병원의 forward/backward를 진짜 HTTP로 호출한다. 병원 간 통신
    시간(peer_comm)을 epoch마다 재서 로그로 남긴다 — /train은 한 프로세스
    시뮬레이션이라 이 오버헤드가 원천적으로 존재하지 않았던 것과 대조된다.
    """
    with _train_lock:
        n_hospitals = len(SLEEP_FEATURE_GROUPS)
        peer_ids    = [i for i in range(n_hospitals) if i != HOSPITAL_ID]
        print(f"[Hospital {HOSPITAL_ID}] (entry-point) HTTP 분산 학습 시작 — peers={peer_ids}")

        t_data_start = time.perf_counter()
        X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
        partitions, _, _, y_train_raw, _ = partition_data_vertical(
            X, y, num_clients=n_hospitals, feature_groups=SLEEP_FEATURE_GROUPS
        )
        my_X_train = torch.tensor(partitions[HOSPITAL_ID]["X_train"], dtype=torch.float32)
        y_train    = torch.tensor(y_train_raw, dtype=torch.float32)
        n          = len(y_train)
        print(f"  [timing] CSV 로드+전처리: {(time.perf_counter()-t_data_start)*1000:.1f} ms")

        total_emb   = emb_dim * n_hospitals
        shared_W    = nn.Linear(total_emb, 1)
        my_hospital = HospitalModel(HOSPITAL_ID, SLEEP_FEATURE_GROUPS, emb_dim, shared_W)
        optimizer   = torch.optim.Adam(
            list(my_hospital.sub.parameters()) + list(shared_W.parameters()), lr=1e-3
        )
        beaver = BeaverProvider(n_hospitals)

        async with httpx.AsyncClient(timeout=60.0) as client:
            for peer_id in peer_ids:
                r = await client.post(f"{_peer_url(peer_id)}/train_init", json={"emb_dim": emb_dim})
                r.raise_for_status()
        print(f"[Hospital {HOSPITAL_ID}] (entry-point) 모든 peer /train_init 완료")

        t_train_start   = time.perf_counter()
        comm_ms_total   = 0.0

        for epoch in range(1, n_epochs + 1):
            perm          = torch.randperm(n)
            epoch_loss    = 0.0
            n_batches     = 0
            comm_ms_epoch = 0.0
            t_epoch_start = time.perf_counter()

            async with httpx.AsyncClient(timeout=60.0) as client:
                for start in range(0, n, batch_size):
                    idx         = perm[start : start + batch_size]
                    y_batch     = y_train[idx]
                    indices     = idx.tolist()
                    batch_token = str(uuid.uuid4())
                    optimizer.zero_grad()

                    # 이 병원(entry-point) 몫은 로컬 forward — 통신 없음
                    my_emb = my_hospital.local_emb(my_X_train[idx])

                    # peer들에게는 실제 HTTP로 forward 요청 — 병원 간 통신 발생 지점
                    peer_embs = {}
                    for peer_id in peer_ids:
                        t0 = time.perf_counter()
                        r = await client.post(
                            f"{_peer_url(peer_id)}/train_forward",
                            json={"batch_token": batch_token, "indices": indices},
                        )
                        r.raise_for_status()
                        comm_ms_epoch += (time.perf_counter() - t0) * 1000
                        peer_embs[peer_id] = torch.tensor(
                            r.json()["emb"], dtype=torch.float32, requires_grad=True
                        )

                    local_embs = [None] * n_hospitals
                    local_embs[HOSPITAL_ID] = my_emb
                    for peer_id, emb in peer_embs.items():
                        local_embs[peer_id] = emb

                    if dp_sigma > 0:
                        for emb in local_embs:
                            emb.register_hook(lambda g: g + torch.randn_like(g) * dp_sigma)

                    all_shares = [additive_split(emb, n=n_hospitals) for emb in local_embs]
                    concat_shares = []
                    for j in range(n_hospitals):
                        received = [
                            apply_dp_noise(all_shares[i][j], dp_sigma) for i in range(n_hospitals)
                        ]
                        concat_shares.append(torch.cat(received, dim=1))

                    logit_shares = [
                        my_hospital.logit_share(concat_shares[j], j == 0)
                        for j in range(n_hospitals)
                    ]

                    triple1 = beaver.generate_triple(logit_shares[0].shape)
                    triple2 = beaver.generate_triple(logit_shares[0].shape)
                    pred_shares = beaver.sigmoid_approx(logit_shares, triple1, triple2)

                    y_shares    = additive_split(y_batch.unsqueeze(1), n=n_hospitals)
                    diff_shares = [pred_shares[i] - y_shares[i] for i in range(n_hospitals)]
                    diff        = sum(diff_shares)
                    loss        = (diff ** 2).mean()
                    loss.backward()
                    optimizer.step()

                    # peer들에게는 gradient를 HTTP로 전송 — 이것도 병원 간 통신
                    for peer_id in peer_ids:
                        t0 = time.perf_counter()
                        r = await client.post(
                            f"{_peer_url(peer_id)}/train_backward",
                            json={"batch_token": batch_token, "grad": peer_embs[peer_id].grad.tolist()},
                        )
                        r.raise_for_status()
                        comm_ms_epoch += (time.perf_counter() - t0) * 1000

                    epoch_loss += loss.item()
                    n_batches  += 1

            epoch_wall_ms = (time.perf_counter() - t_epoch_start) * 1000
            comm_ms_total += comm_ms_epoch

            if epoch % 5 == 0 or epoch == 1:
                pct = (comm_ms_epoch / epoch_wall_ms * 100) if epoch_wall_ms > 0 else 0.0
                print(
                    f"  [Hospital {HOSPITAL_ID}] Epoch {epoch:3d}/{n_epochs} | "
                    f"loss={epoch_loss/n_batches:.4f} | wall={epoch_wall_ms:.1f}ms | "
                    f"peer_comm={comm_ms_epoch:.1f}ms ({pct:.1f}% of wall, {n_batches} batches)"
                )

        train_total_s = time.perf_counter() - t_train_start
        comm_pct = (comm_ms_total / 1000 / train_total_s * 100) if train_total_s > 0 else 0.0
        print(
            f"[Hospital {HOSPITAL_ID}] (entry-point) HTTP 분산 학습 완료 — "
            f"총 {train_total_s:.1f}s, 병원간 통신 누적 {comm_ms_total/1000:.1f}s ({comm_pct:.1f}%)"
        )

        ckpt = {
            "mode":           "distributed",
            "feature_groups": SLEEP_FEATURE_GROUPS,
            "emb_dim":        emb_dim,
            f"sub_{HOSPITAL_ID}": my_hospital.sub.state_dict(),
            "top_W":          shared_W.state_dict(),
            "scaler_mean":    scaler.mean_.tolist(),
            "scaler_scale":   scaler.scale_.tolist(),
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            for peer_id in peer_ids:
                r = await client.get(f"{_peer_url(peer_id)}/train_export")
                r.raise_for_status()
                ckpt[f"sub_{peer_id}"] = torch.load(io.BytesIO(r.content), weights_only=True)

        torch.save(ckpt, MODEL_PATH)
        _load_model()
        print(f"[Hospital {HOSPITAL_ID}] (entry-point) 체크포인트 저장 및 재로드 완료.")


@app.post("/train_distributed", response_model=TrainResponse)
async def train_distributed(background_tasks: BackgroundTasks):
    """/train과 달리 진짜로 세 컨테이너가 HTTP로 통신하며 학습한다. 이 요청을
    받은 병원이 그 세션 동안만 진입점이 되어(고정된 역할 아님, /infer와 동일한
    패턴) peer 두 병원의 /train_forward, /train_backward를 실제로 호출한다.
    컨테이너 로그에 epoch별 병원 간 통신 시간이 찍힌다."""
    if CSV_PATH is None:
        raise HTTPException(400, "CSV_PATH env var not set. Real SHHS CSV is required for training.")
    if _train_lock.locked():
        return TrainResponse(status="running", message="Training already in progress.")
    background_tasks.add_task(_run_training_http_distributed, csv_path=CSV_PATH)
    return TrainResponse(
        status="started",
        message="HTTP-distributed training started (this hospital is the entry-point for this session). Check container logs for peer communication overhead.",
    )


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
