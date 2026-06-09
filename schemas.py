from pydantic import BaseModel
from typing import Dict, Optional


class LogitShareRequest(BaseModel):
    enc_xi_b64: str               # base64: this hospital's enc(features_i)
    he_ctx_b64: Optional[str] = None

class LogitShareResponse(BaseModel):
    enc_logit_share_b64: str      # base64: enc(logit_i) from this hospital


class InferRequest(BaseModel):
    enc_xi_b64: Dict[str, str]    # {"0": b64, "1": b64, "2": b64}
    he_ctx_b64: Optional[str] = None

class InferResponse(BaseModel):
    enc_logit_b64: str            # Watch decrypts this with secret key


class TrainResponse(BaseModel):
    status: str                   # "started" | "running" | "done"
    message: str
