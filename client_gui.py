# 임의 파일 - client , galaxy watch
import base64
import math
import threading
import time                                            # client-hospital / 병원간 통신 시간 측정
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
        self.geometry("780x1150")
        self.resizable(False, False)

        self.entries        = {}
        self.context        = None
        self.normalized     = None
        self._busy          = False
        self.entry_choice   = tk.StringVar(value="Hospital 0")

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

        tk.Button(button_frame, text="3. Send to Hospitals (direct)",
                  command=lambda: self._run_in_thread(self._send_task), width=24).grid(
            row=1, column=1, padx=5, pady=4)

        tk.Button(button_frame, text="4. Send via /infer (entry-point)",
                  command=lambda: self._run_in_thread(self._send_infer_task), width=24).grid(
            row=2, column=0, padx=5, pady=4)

        entry_frame = tk.Frame(button_frame)
        entry_frame.grid(row=2, column=1, padx=5, pady=4)
        tk.Label(entry_frame, text="Entry-point hospital:").pack(side="left")
        tk.OptionMenu(entry_frame, self.entry_choice,
                      "Hospital 0", "Hospital 1", "Hospital 2").pack(side="left")

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

        tk.Label(self, text="Timing Breakdown (client↔hospital / hospital↔hospital)",
                 font=("Arial", 11, "bold")).pack(pady=(8, 2))
        self.timing_text = tk.Text(self, height=10, width=98)
        self.timing_text.pack(pady=5)

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

    def _clear_timing(self):
        self.after(0, self.timing_text.delete, "1.0", "end")

    def _log_timing(self, text):
        print(f"[timing] {text}")                      # 터미널에서도 바로 확인 가능하도록
        self.after(0, self.timing_text.insert, "end", text + "\n")

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

    # ── Phase 3: 각 병원에 직접 전송 → 합산 → 복호화 (client↔hospital만, 병원간 통신 없음) ───

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

        self._clear_timing()
        self._log_timing("=== Direct pattern (client -> each hospital) ===")

        try:
            t_flow_start = time.perf_counter()

            # 컨텍스트를 각 병원에 먼저 업로드 — raw bytes multipart (JSON 불안정 문제 회피)
            for i, url in enumerate(HOSPITAL_URLS):
                self._set_status(f"Status: 병원 {i} 컨텍스트 업로드 중...")
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{url}/upload_context",
                    data=ctx_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=300,
                )
                resp.raise_for_status()
                self._log_timing(f"upload_context -> hospital {i}: {(time.perf_counter()-t0)*1000:.1f} ms")

            enc_logit = None
            for i, (url, indices) in enumerate(zip(HOSPITAL_URLS, FEATURE_GROUPS)):
                self._set_status(f"Status: 병원 {i} ({url}) 추론 요청 중...")

                x_slice  = [self.normalized[j] for j in indices]
                pad      = _next_pow2(len(x_slice))
                x_padded = x_slice + [0.0] * (pad - len(x_slice))
                enc_xi   = ts.ckks_vector(self.context, x_padded)
                xi_b64   = base64.b64encode(enc_xi.serialize()).decode()

                t0 = time.perf_counter()
                resp = requests.post(
                    f"{url}/compute_logit_share",
                    json={"enc_xi_b64": xi_b64},
                    timeout=300,
                )
                resp.raise_for_status()
                self._log_timing(f"compute_logit_share -> hospital {i}: {(time.perf_counter()-t0)*1000:.1f} ms  (client<->hospital)")

                enc_l = ts.ckks_vector_from(
                    self.context,
                    base64.b64decode(resp.json()["enc_logit_share_b64"]),
                )
                enc_logit = enc_l if enc_logit is None else enc_logit + enc_l

            logit = enc_logit.decrypt()[0]
            prob  = sigmoid(logit)
            self._log_timing(f"--- total client-side flow: {(time.perf_counter()-t_flow_start)*1000:.1f} ms ---")
            self._log_timing("(이 패턴은 병원간 통신이 없음 — 클라이언트가 세 병원 결과를 직접 합산)")

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

    # ── Phase 4: entry-point 병원 하나로 전송 → 병원간 통신으로 합산 → 복호화 ───

    def _send_infer_task(self):
        if ts is None:
            self.after(0, messagebox.showerror, "TenSEAL error",
                       f"TenSEAL not available.\n{IMPORT_ERROR}")
            return
        if self.context is None or self.normalized is None:
            self._encrypt_task()
        if self.normalized is None:
            return

        entry_id  = int(self.entry_choice.get().split()[-1])
        entry_url = HOSPITAL_URLS[entry_id]

        pub_ctx = self.context.copy()
        pub_ctx.make_context_public() 
        ctx_bytes = pub_ctx.serialize(
            save_public_key=True, save_secret_key=False,
            save_galois_keys=True, save_relin_keys=True,
        )

        self._clear_timing()
        self._log_timing(f"=== Entry-point pattern (client -> hospital {entry_id} -> peers) ===")

        try:
            t_flow_start = time.perf_counter()

            # 세 병원 모두 같은 컨텍스트를 알아야 암호문을 서로 더할 수 있음 
            for i, url in enumerate(HOSPITAL_URLS):
                self._set_status(f"Status: 병원 {i} 컨텍스트 업로드 중...")
                t0 = time.perf_counter()
                resp = requests.post( #  context 업로드
                    f"{url}/upload_context", 
                    data=ctx_bytes, 
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=300,
                )
                resp.raise_for_status()
                self._log_timing(f"upload_context -> hospital {i}: {(time.perf_counter()-t0)*1000:.1f} ms")

            enc_xi_b64 = {}
            for i, indices in enumerate(FEATURE_GROUPS):
                x_slice  = [self.normalized[j] for j in indices]
                pad      = _next_pow2(len(x_slice))
                x_padded = x_slice + [0.0] * (pad - len(x_slice)) 
                enc_xi   = ts.ckks_vector(self.context, x_padded) 
                enc_xi_b64[str(i)] = base64.b64encode(enc_xi.serialize()).decode()  # 바이너리코드를 json 
                # 에 넣기 위해 base64 인코딩 -> ascii 로 변환 

            self._set_status(f"Status: entry-point 병원 {entry_id} ({entry_url}) 에 /infer 요청 중...")
            t0 = time.perf_counter()
            # 추론할 때 json 데이터에 암호문을 넣음  
            resp = requests.post(
                f"{entry_url}/infer",
                json={"enc_xi_b64": enc_xi_b64},  
                timeout=300,
            )
            resp.raise_for_status()
            client_to_entry_ms = (time.perf_counter() - t0) * 1000
            body = resp.json()

            self._log_timing(f"client -> entry hospital {entry_id} (/infer round trip): {client_to_entry_ms:.1f} ms  (client<->hospital)")

            timing = body["timing"]
            self._log_timing(f"  entry hospital local HE compute: {timing['local_compute_ms']:.1f} ms")
            for peer_id, ms in timing["peer_calls_ms"].items():
                self._log_timing(f"  entry hospital {entry_id} -> peer hospital {peer_id}: {ms:.1f} ms  (hospital<->hospital)")
            self._log_timing(f"  entry hospital total server-side handling: {timing['total_ms']:.1f} ms")

            # 마지막 합산 단계 
            # entry-point는 합산을 안 하고 각자의 enc(logit_share)만 릴레이함 —
            # 최종 합산+복호화는 direct 패턴과 동일하게 client가 로컬에서 함
            enc_logit = None
            for share_b64 in body["enc_logit_shares_b64"].values():
                enc_l = ts.ckks_vector_from(self.context, base64.b64decode(share_b64))
                enc_logit = enc_l if enc_logit is None else enc_logit + enc_l
            logit = enc_logit.decrypt()[0]
            prob  = sigmoid(logit)
            self._log_timing(f"--- total client-side flow: {(time.perf_counter()-t_flow_start)*1000:.1f} ms ---")

            def update_ui():
                self.result.config(
                    text=f"logit: {logit:.4f}   |   수면무호흡 위험도: {prob*100:.2f}%  (entry-point: hospital {entry_id})"
                )
                self.status.config(text="Status: entry-point 병원 응답 수신 → 로컬 복호화 완료.")

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
