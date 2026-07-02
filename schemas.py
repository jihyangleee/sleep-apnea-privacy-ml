from pydantic import BaseModel


class LogitShareRequest(BaseModel):
    enc_xi_b64: str               # base64: this hospital's enc(features_i)

class LogitShareResponse(BaseModel):
    enc_logit_share_b64: str      # base64: enc(logit_i) from this hospital


class TrainResponse(BaseModel):
    status: str                   # "started" | "running" | "done"
    message: str
