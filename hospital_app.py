"""FastAPI server for one hospital node.

Start with:
    HOSPITAL_ID=0 uvicorn hospital_app:app --port 8001
    HOSPITAL_ID=1 uvicorn hospital_app:app --port 8002
    HOSPITAL_ID=2 uvicorn hospital_app:app --port 8003

Requires a trained checkpoint (vertical_model.pt):
    python main.py --mode distributed

Or trigger training via POST /train after starting the servers.
"""

import os
import base64
import threading
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import tenseal as ts
from fastapi import FastAPI, HTTPException, BackgroundTasks

from schemas import (
    LogitShareRequest, LogitShareResponse,
    TrainResponse,
)
from model import HospitalModel
from he_client import build_he_context
from simulate import SLEEP_FEATURE_GROUPS, run_distributed_simulation

# ── Configuration from environment ───────────────────────────────────────────

HOSPITAL_ID = int(os.environ.get("HOSPITAL_ID", "0"))
MODEL_PATH  = os.environ.get("MODEL_PATH", "vertical_model.pt")
CSV_PATH    = os.environ.get("CSV_PATH") or None

# ── State ─────────────────────────────────────────────────────────────────────

hospital: HospitalModel | None = None
he_ctx:   ts.Context    | None = None
_train_lock = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    global hospital, he_ctx

    ckpt      = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    fg        = ckpt["feature_groups"]
    emb_dim   = ckpt["emb_dim"]
    n         = len(fg)
    total_emb = emb_dim * n

    shared_W = nn.Linear(total_emb, 1)
    shared_W.load_state_dict(ckpt["top_W"])

    h = HospitalModel(HOSPITAL_ID, fg, emb_dim, shared_W)
    h.sub.load_state_dict(ckpt[f"sub_{HOSPITAL_ID}"])
    h.eval()

    # Each hospital holds its own column slice of the top linear weight
    W_top   = shared_W.weight.detach().numpy()   # (1, total_emb)
    W_top_i = W_top[:, HOSPITAL_ID * emb_dim : (HOSPITAL_ID + 1) * emb_dim]  # (1, emb_dim)
    b_top   = shared_W.bias.detach().numpy()     # (1,)

    h.build_he_weights(W_top_i, b_top)

    hospital = h
    he_ctx   = build_he_context()
    print(f"[Hospital {HOSPITAL_ID}] Model loaded from {MODEL_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        _load_model()
    else:
        print(f"[Hospital {HOSPITAL_ID}] No checkpoint at {MODEL_PATH}. POST /train to train first.")
    yield


def _require_model():
    if hospital is None:
        raise HTTPException(503, "Model not loaded. POST /train first.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title=f"Hospital {HOSPITAL_ID}", lifespan=lifespan)


# ── Inter-hospital endpoint ───────────────────────────────────────────────────

def _resolve_ctx(he_ctx_b64: str | None) -> ts.Context:
    if he_ctx_b64:
        return ts.context_from(base64.b64decode(he_ctx_b64))
    return he_ctx


@app.post("/compute_logit_share", response_model=LogitShareResponse)
async def compute_logit_share(req: LogitShareRequest):
    """enc(features_i) → enc(emb_i) → enc(logit_i) in one shot.

    Patient (Watch) calls this directly on each hospital with that hospital's
    own encrypted feature slice. No coordinator or inter-hospital communication.

    Hospital 0 adds b_top to its ciphertext so the patient can simply sum all
    three responses to get enc(logit_final), then decrypt with their secret key.
    """
    _require_model()
    ctx     = _resolve_ctx(req.he_ctx_b64)
    enc_xi  = base64.b64decode(req.enc_xi_b64)
    enc_emb = hospital.compute_sub_emb_he(enc_xi, ctx)
    enc_logit_share = hospital.compute_logit_share_he(enc_emb, ctx)

    # Hospital 0 absorbs b_top into its ciphertext so the patient just sums
    if HOSPITAL_ID == 0 and hospital._b_top is not None:
        enc_l = ts.lazy_ckks_vector_from(enc_logit_share)
        enc_l.link_context(ctx)
        enc_l += hospital._b_top
        enc_logit_share = enc_l.serialize()

    return LogitShareResponse(
        enc_logit_share_b64=base64.b64encode(enc_logit_share).decode()
    )


# ── Training endpoint ─────────────────────────────────────────────────────────

def _run_training(csv_path=None, dreamt_dir=None, dp_sigma=0.01, n_epochs=30):
    with _train_lock:
        hospitals_tr, W, scaler = run_distributed_simulation(
            csv_path=csv_path, dreamt_dir=dreamt_dir,
            n_epochs=n_epochs, dp_sigma=dp_sigma,
        )
        torch.save(
            {
                "mode":           "distributed",
                "feature_groups": SLEEP_FEATURE_GROUPS,
                "emb_dim":        hospitals_tr[0].emb_dim,
                "sub_0":          hospitals_tr[0].sub.state_dict(),
                "sub_1":          hospitals_tr[1].sub.state_dict(),
                "sub_2":          hospitals_tr[2].sub.state_dict(),
                "top_W":          W.state_dict(),
                "scaler_mean":    scaler.mean_.tolist(),
                "scaler_scale":   scaler.scale_.tolist(),
            },
            MODEL_PATH,
        )
        _load_model()
        print(f"[Hospital {HOSPITAL_ID}] Retraining complete.")


@app.post("/train", response_model=TrainResponse)
async def train(background_tasks: BackgroundTasks):
    """Trigger distributed FL training. Saves checkpoint and reloads model."""
    if _train_lock.locked():
        return TrainResponse(status="running", message="Training already in progress.")
    background_tasks.add_task(_run_training, csv_path=CSV_PATH)
    return TrainResponse(status="started", message="Training started in background. GET /health to check.")


@app.post("/reload", response_model=TrainResponse)
async def reload_model():
    """Reload model from checkpoint (useful after another hospital triggers training)."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(404, f"No checkpoint at {MODEL_PATH}")
    _load_model()
    return TrainResponse(status="done", message=f"Model reloaded from {MODEL_PATH}")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "hospital_id":  HOSPITAL_ID,
        "model_loaded": hospital is not None,
        "model_path":   MODEL_PATH,
    }
