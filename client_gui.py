import base64
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import requests

try:
    from feature_processor import extract_features
except Exception:
    extract_features = None

try:
    import tenseal as ts
except Exception as exc:
    ts = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


# ── 병원 서버 주소 (각 병원에 직접 연결) ─────────────────────────────────────
HOSPITAL_URLS = [
    "http://172.24.7.163:8001",
    "http://172.24.7.163:8002",
    "http://172.24.7.163:8003",
]
# galaxy watch 에서 요청하는 부분 - client 파트
# 병원별 feature 슬라이스 인덱스 (simulate.py의 SLEEP_FEATURE_GROUPS와 동일)
# SHHS feature 순서: spo2, avg_hr, slptime, slp_eff, timest34p, age, sex, bmi
FEATURE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]

# GUI 입력 필드 (SHHS feature 순서에 맞춤)
FEATURES = [
    ("SpO2 (%)",                "spo2",       95.0),
    ("Avg heart rate (bpm)",    "avg_hr",     72.0),
    ("Total sleep time (min)",  "slptime",   420.0),
    ("Sleep efficiency (0~1)",  "slp_eff",    0.85),
    ("Deep sleep ratio (0~1)",  "timest34p",  0.20),
    ("Age",                     "age",        50.0),
    ("Sex (male=1, female=0)",  "sex",         1.0),
    ("BMI",                     "bmi",        25.0),
]

DEMO_SHOW_SECRET_KEY = True


def normalize(values):
    spo2, avg_hr, slptime, slp_eff, timest34p, age, sex, bmi = values
    return [
        (spo2     - 95.0)  /  3.0,
        (avg_hr   - 70.0)  / 15.0,
        (slptime  - 420.0) / 90.0,
        (slp_eff  -  0.85) /  0.10,
        (timest34p - 0.20) /  0.10,
        (age      - 50.0)  / 15.0,
        sex,
        (bmi      - 25.0)  /  5.0,
    ]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return max(p, 4)


class SecureHealthClient(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Secure Health HE Client")
        self.geometry("780x920")
        self.resizable(False, False)

        self.entries        = {}
        self.context        = None
        self.normalized     = None
        self._busy          = False

        title = tk.Label(self, text="Encrypted Sleep-Apnea Inference Client",
                         font=("Arial", 15, "bold"))
        title.pack(pady=10)

        if ts is None:
            tk.Label(self, text=f"TenSEAL import failed: {IMPORT_ERROR}",
                     fg="red", wraplength=740).pack(pady=5)

        form = tk.Frame(self)
        form.pack(pady=5)
        for i, (label, key, default) in enumerate(FEATURES):
            tk.Label(form, text=label, anchor="w", width=30).grid(
                row=i, column=0, padx=6, pady=4)
            ent = tk.Entry(form, width=20)
            ent.insert(0, str(default))
            ent.grid(row=i, column=1, padx=6, pady=4)
            self.entries[key] = ent

        button_frame = tk.Frame(self)
        button_frame.pack(pady=8)

        tk.Button(button_frame, text="Load Watch Data",
                  command=self.load_watch_data, width=24).grid(
            row=0, column=0, padx=5, pady=4)

        tk.Button(button_frame, text="1. Generate CKKS Keys",
                  command=lambda: self._run_in_thread(self._generate_context_task), width=24).grid(
            row=0, column=1, padx=5, pady=4)

        tk.Button(button_frame, text="2. Encrypt & Preview",
                  command=lambda: self._run_in_thread(self._encrypt_task), width=24).grid(
            row=1, column=0, padx=5, pady=4)

        tk.Button(button_frame, text="3. Send to Hospitals",
                  command=lambda: self._run_in_thread(self._send_task), width=24).grid(
            row=1, column=1, padx=5, pady=4)

        self.status = tk.Label(self, text="Status: ready", anchor="w",
                               justify="left", wraplength=740)
        self.status.pack(pady=6)

        self.result = tk.Label(self, text="", font=("Arial", 13, "bold"),
                               wraplength=740)
        self.result.pack(pady=6)

        tk.Label(self, text="CKKS Key Information",
                 font=("Arial", 11, "bold")).pack(pady=(8, 2))
        self.key_text = tk.Text(self, height=8, width=98)
        self.key_text.pack(pady=5)

        tk.Label(self, text="Encrypted Feature Preview",
                 font=("Arial", 11, "bold")).pack(pady=(8, 2))
        self.encrypted_text = tk.Text(self, height=12, width=98)
        self.encrypted_text.pack(pady=5)

    # ── 스레드 실행 헬퍼 ──────────────────────────────────────────────────────

    def _run_in_thread(self, task):
        if self._busy:
            self._set_status("Status: 이전 작업이 진행 중입니다.")
            return
        self._busy = True
        threading.Thread(target=self._task_wrapper, args=(task,), daemon=True).start()

    def _task_wrapper(self, task):
        try:
            task()
        except Exception as exc:
            self.after(0, messagebox.showerror, "Error", str(exc))
            self.after(0, self._set_status, f"Status: failed — {exc}")
        finally:
            self._busy = False

    def _set_status(self, text):
        self.after(0, self.status.config, {"text": text})

    # ── helpers ──────────────────────────────────────────────────────────────

    def read_values(self):
        return [float(self.entries[key].get().strip()) for _, key, _ in FEATURES]

    def load_watch_data(self):
        if extract_features is None:
            messagebox.showerror("Watch data error", "feature_processor.py를 찾을 수 없습니다.")
            return
        path = filedialog.askopenfilename(
            title="Select watch data file",
            filetypes=[("Watch data files", "*.csv *.json"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            feats = extract_features(path)
            mapping = {
                "spo2":      "spo2",
                "avg_hr":    "avg_hr",
                "slptime":   "total_sleep_time",
                "slp_eff":   "sleep_efficiency",
                "timest34p": "deep_sleep_ratio",
                "age":       "age",
                "sex":       "sex",
                "bmi":       "bmi",
            }
            for gui_key, feat_key in mapping.items():
                if feat_key in feats:
                    self.entries[gui_key].delete(0, "end")
                    self.entries[gui_key].insert(0, str(feats[feat_key]))
            self.status.config(text=f"Status: loaded watch data from {path}")
        except Exception as e:
            messagebox.showerror("Watch data error", str(e))

    # ── Phase 1: 키 생성 (백그라운드) ────────────────────────────────────────

    def _generate_context_task(self):
        if ts is None:
            self.after(0, messagebox.showerror, "TenSEAL error",
                       f"TenSEAL not available.\n{IMPORT_ERROR}")
            return

        self._set_status("Status: CKKS 키 생성 중... (수초 소요)")

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60],
        )
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()

        pub_ctx = self.context.copy()
        pub_ctx.make_context_public()
        pub_b64 = base64.b64encode(
            pub_ctx.serialize(save_public_key=True, save_secret_key=False,
                              save_galois_keys=True, save_relin_keys=True)
        ).decode()

        def update_ui():
            self.key_text.delete("1.0", "end")
            self.key_text.insert("end", "[CKKS Parameters]\n")
            self.key_text.insert("end", "poly_modulus_degree : 16384\n")
            self.key_text.insert("end", "coeff_mod_bit_sizes : [60,40,40,40,40,40,40,40,60]\n")
            self.key_text.insert("end", "global_scale        : 2^40\n\n")
            self.key_text.insert("end", f"[Public Context Preview]\n{pub_b64[:800]}...\n")
            self.status.config(text="Status: CKKS context generated (poly_modulus_degree=16384).")

        self.after(0, update_ui)

    # ── Phase 2: 암호화 미리보기 (백그라운드) ────────────────────────────────

    def _encrypt_task(self):
        if ts is None:
            self.after(0, messagebox.showerror, "TenSEAL error",
                       f"TenSEAL not available.\n{IMPORT_ERROR}")
            return
        if self.context is None:
            self._generate_context_task()

        self._set_status("Status: 암호화 중...")

        raw             = self.read_values()
        self.normalized = normalize(raw)

        lines = []
        lines.append("[Plain Feature Vector]\n")
        lines.append(str(raw) + "\n\n")
        lines.append("[Normalized Feature Vector]\n")
        lines.append(str(self.normalized) + "\n\n")

        for i, indices in enumerate(FEATURE_GROUPS):
            x_slice  = [self.normalized[j] for j in indices]
            pad      = _next_pow2(len(x_slice))
            x_padded = x_slice + [0.0] * (pad - len(x_slice))
            enc      = ts.ckks_vector(self.context, x_padded)
            b64      = base64.b64encode(enc.serialize()).decode()
            lines.append(f"[Hospital {i} — features {indices}]\n{b64[:300]}...\n\n")

        def update_ui():
            self.encrypted_text.delete("1.0", "end")
            for line in lines:
                self.encrypted_text.insert("end", line)
            self.status.config(text="Status: features encrypted. (3 slices, one per hospital)")

        self.after(0, update_ui)

    # ── Phase 3: 각 병원에 직접 전송 → 합산 → 복호화 (백그라운드) ───────────

    def _send_task(self):
        if ts is None:
            self.after(0, messagebox.showerror, "TenSEAL error",
                       f"TenSEAL not available.\n{IMPORT_ERROR}")
            return
        if self.context is None or self.normalized is None:
            self._encrypt_task()
        if self.normalized is None:
            return

        pub_ctx = self.context.copy()
        pub_ctx.make_context_public()
        ctx_bytes = pub_ctx.serialize(
            save_public_key=True, save_secret_key=False,
            save_galois_keys=True, save_relin_keys=True,
        )

        try:
            # 컨텍스트를 각 병원에 먼저 업로드 — raw bytes multipart (JSON 불안정 문제 회피)
            for i, url in enumerate(HOSPITAL_URLS):
                self._set_status(f"Status: 병원 {i} 컨텍스트 업로드 중...")
                resp = requests.post(
                    f"{url}/upload_context",
                    data=ctx_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=300,
                )
                resp.raise_for_status()

            enc_logit = None
            for i, (url, indices) in enumerate(zip(HOSPITAL_URLS, FEATURE_GROUPS)):
                self._set_status(f"Status: 병원 {i} ({url}) 추론 요청 중...")

                x_slice  = [self.normalized[j] for j in indices]
                pad      = _next_pow2(len(x_slice))
                x_padded = x_slice + [0.0] * (pad - len(x_slice))
                enc_xi   = ts.ckks_vector(self.context, x_padded)
                xi_b64   = base64.b64encode(enc_xi.serialize()).decode()

                resp = requests.post(
                    f"{url}/compute_logit_share",
                    json={"enc_xi_b64": xi_b64},
                    timeout=300,
                )
                resp.raise_for_status()

                enc_l = ts.ckks_vector_from(
                    self.context,
                    base64.b64decode(resp.json()["enc_logit_share_b64"]),
                )
                enc_logit = enc_l if enc_logit is None else enc_logit + enc_l

            logit = enc_logit.decrypt()[0]
            prob  = sigmoid(logit)

            def update_ui():
                self.result.config(
                    text=f"logit: {logit:.4f}   |   수면무호흡 위험도: {prob*100:.2f}%"
                )
                self.status.config(text="Status: 3개 병원 응답 수신 → 로컬 복호화 완료.")

            self.after(0, update_ui)

        except requests.exceptions.Timeout:
            self.after(0, messagebox.showerror, "Timeout", "병원 서버 응답 시간 초과 (300s).")
            self._set_status("Status: timeout.")
        except Exception as exc:
            self.after(0, messagebox.showerror, "Prediction failed", str(exc))
            self._set_status(f"Status: failed — {exc}")


if __name__ == "__main__":
    app = SecureHealthClient()
    app.mainloop()
