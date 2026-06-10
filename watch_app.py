"""Galaxy Watch 역할을 하는 웹 클라이언트.

[추론 프로토콜 — 코디네이터 없음]
  1. 환자 디바이스가 pk/sk 생성 후 특성 암호화
  2. 각 병원에 자신의 feature slice만 직접 전송 (병렬)
  3. 각 병원은 독립 연산 → enc(logit_i) 반환
     (병원 0은 자신의 응답에 b_top 포함)
  4. 환자가 세 암호문을 HE 덧셈으로 합산 → enc(logit_final)
  5. 환자가 sk로 복호화 → 평문 logit
  6. 환자 디바이스에서 정확한 sigmoid 적용 → 위험도

실행:
    pip install fastapi uvicorn httpx jinja2 python-multipart
    uvicorn watch_app:app --host 0.0.0.0 --port 9000
"""

import os
import base64
import asyncio
import math

import numpy as np
import httpx
import tenseal as ts
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

# ── 병원 서버 주소 (직접 접속, 코디네이터 불필요) ─────────────────────────────
HOSPITAL_URLS = [
    os.environ.get("HOSPITAL_A_URL", "http://localhost:8001"),
    os.environ.get("HOSPITAL_B_URL", "http://localhost:8002"),
    os.environ.get("HOSPITAL_C_URL", "http://localhost:8003"),
]

# Feature 슬라이스 인덱스 (simulate.py의 SLEEP_FEATURE_GROUPS와 동일)
FEATURE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]

# ── HE context: Watch만 secret key 보유 ──────────────────────────────────────
def _build_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    return ctx

_patient_ctx    = _build_context()                               # secret key 포함
_public_ctx_b64 = base64.b64encode(
    _patient_ctx.serialize(save_secret_key=False)               # 병원에 전달할 공개 context
).decode()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def encrypt_slices(x: np.ndarray) -> dict:
    """병원별 feature slice를 next-power-of-2 크기로 zero-pad 후 암호화.

    각 병원은 자기 feature만 수신 (VFL 원칙 유지).
    Zero-pad는 TenSEAL mm_의 소형 벡터 오작동을 방지.
    """
    result = {}
    for hospital_id, indices in enumerate(FEATURE_GROUPS):
        raw          = x[indices].astype(np.float64)
        padded_size  = max(_next_pow2(len(raw)), 4)
        padded       = np.zeros(padded_size)
        padded[:len(raw)] = raw
        vec          = ts.ckks_vector(_patient_ctx, padded.tolist())
        result[str(hospital_id)] = base64.b64encode(vec.serialize()).decode()
    return result


def decrypt_and_sigmoid(enc_logit_b64: str) -> float:
    """환자 sk로 복호화 후 정확한 sigmoid 적용."""
    enc = ts.lazy_ckks_vector_from(base64.b64decode(enc_logit_b64))
    enc.link_context(_patient_ctx)
    logit = enc.decrypt()[0]
    return 1.0 / (1.0 + math.exp(-logit))


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────

app = FastAPI(title="Galaxy Watch Client")

_FEATURE_META = [
    ("spo2",       "SpO2 산소포화도 (%)",       "병원 A", 95.0),
    ("avg_hr",     "평균 심박수 (bpm)",          "병원 A", 65.0),
    ("slptime",    "총 수면시간 (분)",           "병원 B", 420.0),
    ("slp_eff",    "수면 효율 (%)",              "병원 B", 85.0),
    ("timest34p",  "깊은수면 비율 (%)",          "병원 B", 20.0),
    ("age",        "나이",                       "병원 C", 45.0),
    ("gender",     "성별 (0=여, 1=남)",          "병원 C", 1.0),
    ("bmi",        "BMI",                        "병원 C", 25.0),
]


def _render(result: str = "", error: str = "", values: list = None):
    if values is None:
        values = [m[3] for m in _FEATURE_META]

    rows = ""
    for i, (name, label, hospital, _) in enumerate(_FEATURE_META):
        rows += f"""
        <tr>
          <td style="padding:6px 12px;color:#888;font-size:13px">{hospital}</td>
          <td style="padding:6px 12px">{label}</td>
          <td style="padding:6px 4px">
            <input type="number" name="{name}" value="{values[i]}"
                   step="any" required
                   style="width:90px;padding:4px 8px;border:1px solid #ddd;border-radius:4px">
          </td>
        </tr>"""

    hospital_list = "".join(
        f'<li><code>{url}</code></li>' for url in HOSPITAL_URLS
    )

    result_block = ""
    if result:
        prob  = float(result)
        color = "#e74c3c" if prob >= 0.5 else "#27ae60"
        label = "수면무호흡 위험" if prob >= 0.5 else "정상 범위"
        result_block = f"""
        <div style="margin-top:24px;padding:20px;background:#f8f9fa;border-radius:8px;text-align:center">
          <div style="font-size:40px;font-weight:bold;color:{color}">{prob*100:.1f}%</div>
          <div style="font-size:18px;color:{color};margin-top:8px">{label}</div>
          <div style="font-size:12px;color:#aaa;margin-top:12px">
            각 병원에 독립 암호화 연산 → 환자 디바이스에서 HE 합산 및 복호화
          </div>
        </div>"""

    if error:
        result_block = f"""
        <div style="margin-top:16px;padding:12px;background:#fee;border-radius:6px;color:#c00">
          오류: {error}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Galaxy Watch — 수면무호흡 예측</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 560px;
            margin: 40px auto; padding: 0 20px; color: #333; }}
    h2   {{ color: #1a73e8; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th   {{ text-align:left; padding: 6px 12px; color: #555;
            border-bottom: 2px solid #eee; font-size:13px; }}
    button {{ margin-top:20px; width:100%; padding:12px;
              background:#1a73e8; color:#fff; border:none;
              border-radius:6px; font-size:16px; cursor:pointer; }}
    button:hover {{ background:#1558b0; }}
    .hospitals {{ font-size:12px; color:#888; margin-bottom:16px; }}
    .hospitals ul {{ margin:4px 0; padding-left:20px; }}
  </style>
</head>
<body>
  <h2>⌚ Galaxy Watch</h2>
  <div class="hospitals">
    직접 연결 병원 (코디네이터 없음):
    <ul>{hospital_list}</ul>
  </div>

  <form method="post" action="/predict">
    <table>
      <tr><th>병원</th><th>측정값</th><th>수치</th></tr>
      {rows}
    </table>
    <button type="submit">🔐 암호화 후 예측 요청</button>
  </form>
  {result_block}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _render()


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:   Request,
    spo2:      float = Form(...),
    avg_hr:    float = Form(...),
    slptime:   float = Form(...),
    slp_eff:   float = Form(...),
    timest34p: float = Form(...),
    age:       float = Form(...),
    gender:    float = Form(...),
    bmi:       float = Form(...),
):
    values = [spo2, avg_hr, slptime, slp_eff, timest34p, age, gender, bmi]
    x      = np.array(values, dtype=np.float64)

    # ── Phase 1: 환자가 각 병원의 feature slice 암호화 ────────────────────────
    enc_xi = encrypt_slices(x)

    try:
        # ── Phase 2: 각 병원에 직접 병렬 요청 (코디네이터 없음) ──────────────
        async with httpx.AsyncClient(timeout=300.0) as client:
            resps = await asyncio.gather(*[
                client.post(
                    f"{HOSPITAL_URLS[i]}/compute_logit_share",
                    json={
                        "enc_xi_b64":  enc_xi[str(i)],
                        "he_ctx_b64":  _public_ctx_b64,
                    },
                )
                for i in range(len(HOSPITAL_URLS))
            ])

        for i, r in enumerate(resps):
            if r.status_code != 200:
                return HTMLResponse(_render(
                    error=f"병원 {i} 응답 오류 ({r.status_code}): {r.text}",
                    values=values,
                ))

        # ── Phase 3: 환자 디바이스에서 암호문 HE 덧셈 합산 ──────────────────
        # 병원 0의 응답에는 b_top이 이미 포함되어 있으므로 단순 합산만으로 충분
        enc_logit = None
        for r in resps:
            enc_l = ts.lazy_ckks_vector_from(
                base64.b64decode(r.json()["enc_logit_share_b64"])
            )
            enc_l.link_context(_patient_ctx)   # secret key 포함 context로 연결
            enc_logit = enc_l if enc_logit is None else enc_logit + enc_l

        # ── Phase 4: 환자 sk로 복호화 + 정확한 sigmoid 적용 ─────────────────
        enc_logit_b64 = base64.b64encode(enc_logit.serialize()).decode()
        prob          = decrypt_and_sigmoid(enc_logit_b64)

        return HTMLResponse(_render(result=str(prob), values=values))

    except (httpx.ConnectError, httpx.RemoteProtocolError, OSError):
        failed = [
            HOSPITAL_URLS[i] for i, r in enumerate(resps)
            if r.status_code != 200
        ] if 'resps' in dir() else HOSPITAL_URLS
        return HTMLResponse(_render(
            error=f"병원 서버 연결 실패: {failed}. 모든 병원 서버가 실행 중인지 확인하세요.",
            values=values,
        ))
    except Exception as e:
        return HTMLResponse(_render(error=f"오류: {type(e).__name__}: {e}", values=values))
