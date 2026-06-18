from pydantic import BaseModel
from typing import Optional


class LogitShareRequest(BaseModel):
    enc_xi_b64: str               # base64: this hospital's enc(features_i)
    he_ctx_b64: Optional[str] = None

class LogitShareResponse(BaseModel):
    enc_logit_share_b64: str      # base64: enc(logit_i) from this hospital


class TrainResponse(BaseModel):
    status: str                   # "started" | "running" | "done"
    message: str
