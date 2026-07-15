"""
CKKS 기반 수면무호흡 추론 클라이언트 — 개선 버전

핵심 변경사항
1. 사용자별 CKKS private/public context를 파일에 저장하고 재사용한다.
2. 클라이언트가 병원 A/B/C에 각 병원의 feature 암호문만 직접 전송한다.
3. 한 번의 버튼 클릭으로 키 로드 → 입력 검증 → 정규화 → 암호화 →
   context 동기화(필요한 경우만) → 병원별 병렬 추론 → 결과 합산 → 복호화를 수행한다.
4. context/ciphertext/JSON/응답 크기를 측정하고 가상 네트워크 환경별 예상 전송시간을 계산한다.

주의
- private_context.bin에는 Secret Key가 포함되므로 클라이언트 밖으로 절대 전송하지 않는다.
- public_context.bin만 병원 서버에 업로드한다.
- 병원 서버가 재시작되면 context 캐시가 사라질 수 있으므로 GUI의 '컨텍스트 강제 재전송'을 사용한다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


# ── 서버 및 모델 입력 설정 ───────────────────────────────────────────────────
HOSPITAL_URLS = [
    "http://172.24.7.163:8001",
    "http://172.24.7.163:8002",
    "http://172.24.7.163:8003",
]

# SHHS feature 순서: spo2, avg_hr, slptime, slp_eff, timest34p, age, sex, bmi
FEATURE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]

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

# 발표/실험용 가상 네트워크 환경. 필요하면 수치만 수정한다.
NETWORK_SCENARIOS = {
    "LAN": {"bandwidth_mbps": 1000.0, "rtt_ms": 1.0},
    "Campus/Metro": {"bandwidth_mbps": 100.0, "rtt_ms": 20.0},
    "WAN": {"bandwidth_mbps": 20.0, "rtt_ms": 80.0},
}

KEY_ROOT = Path.home() / ".secure_health_he" / "users"
PRIVATE_CONTEXT_NAME = "private_context.bin"
PUBLIC_CONTEXT_NAME = "public_context.bin"
METADATA_NAME = "metadata.json"

REQUEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class UserKeyPaths:
    directory: Path
    private_context: Path
    public_context: Path
    metadata: Path


def normalize(values: list[float]) -> list[float]:
    spo2, avg_hr, slptime, slp_eff, timest34p, age, sex, bmi = values
    return [
        (spo2      - 95.0) / 3.0,
        (avg_hr    - 70.0) / 15.0,
        (slptime   - 420.0) / 90.0,
        (slp_eff   - 0.85) / 0.10,
        (timest34p - 0.20) / 0.10,
        (age       - 50.0) / 15.0,
        sex,
        (bmi       - 25.0) / 5.0,
    ]


def sigmoid(x: float) -> float:
    # 큰 절댓값에서도 overflow가 나지 않는 안정적인 sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(p, 4)


def _safe_user_id(user_id: str) -> str:
    value = user_id.strip()
    if not value:
        raise ValueError("사용자 ID를 입력하세요.")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if safe in {".", ".."}:
        raise ValueError("올바른 사용자 ID를 입력하세요.")
    return safe


def _key_paths(user_id: str) -> UserKeyPaths:
    directory = KEY_ROOT / _safe_user_id(user_id)
    return UserKeyPaths(
        directory=directory,
        private_context=directory / PRIVATE_CONTEXT_NAME,
        public_context=directory / PUBLIC_CONTEXT_NAME,
        metadata=directory / METADATA_NAME,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows에서는 chmod 의미가 제한적이므로 실패해도 저장 자체는 유지한다.
        pass


def _context_id(public_context_bytes: bytes) -> str:
    return hashlib.sha256(public_context_bytes).hexdigest()[:16]


def _json_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.2f} KiB"
    return f"{size / 1024**2:.2f} MiB"


def _estimated_one_way_ms(size_bytes: int, bandwidth_mbps: float) -> float:
    return (size_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0)) * 1000.0


class SecureHealthClient(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Secure Health HE Client")
        self.geometry("830x1120")
        self.resizable(False, False)

        self.entries: dict[str, tk.Entry] = {}
        self.context = None                    # Secret Key가 포함된 client-only context
        self.public_context_bytes: bytes | None = None
        self.current_user_id: str | None = None
        self.current_context_id: str | None = None
        self._busy = False

        # 이 프로세스에서 이미 context를 전송한 병원과 context_id 기록
        self.synced_context_ids: dict[int, str] = {}

        title = tk.Label(
            self,
            text="Encrypted Sleep-Apnea Inference Client",
            font=("Arial", 15, "bold"),
        )
        title.pack(pady=10)

        if ts is None:
            tk.Label(
                self,
                text=f"TenSEAL import failed: {IMPORT_ERROR}",
                fg="red",
                wraplength=790,
            ).pack(pady=5)

        identity_frame = tk.Frame(self)
        identity_frame.pack(pady=4)
        tk.Label(identity_frame, text="User ID", width=18, anchor="e").grid(row=0, column=0, padx=5)
        self.user_id_entry = tk.Entry(identity_frame, width=28)
        self.user_id_entry.insert(0, "user001")
        self.user_id_entry.grid(row=0, column=1, padx=5)
        self.force_context_upload = tk.BooleanVar(value=False)
        tk.Checkbutton(
            identity_frame,
            text="컨텍스트 강제 재전송",
            variable=self.force_context_upload,
        ).grid(row=0, column=2, padx=10)

        form = tk.Frame(self)
        form.pack(pady=5)
        for i, (label, key, default) in enumerate(FEATURES):
            tk.Label(form, text=label, anchor="w", width=30).grid(
                row=i, column=0, padx=6, pady=4
            )
            ent = tk.Entry(form, width=20)
            ent.insert(0, str(default))
            ent.grid(row=i, column=1, padx=6, pady=4)
            self.entries[key] = ent

        button_frame = tk.Frame(self)
        button_frame.pack(pady=8)

        tk.Button(
            button_frame,
            text="Load Watch Data",
            command=self.load_watch_data,
            width=25,
        ).grid(row=0, column=0, padx=5, pady=4)

        tk.Button(
            button_frame,
            text="Create/Replace User Keys",
            command=lambda: self._run_in_thread(self._create_user_keys_task),
            width=25,
        ).grid(row=0, column=1, padx=5, pady=4)

        tk.Button(
            button_frame,
            text="Run Secure Inference",
            command=lambda: self._run_in_thread(self._automatic_inference_task),
            width=54,
            height=2,
        ).grid(row=1, column=0, columnspan=2, padx=5, pady=6)

        self.status = tk.Label(
            self,
            text="Status: ready",
            anchor="w",
            justify="left",
            wraplength=790,
        )
        self.status.pack(pady=6)

        self.result = tk.Label(
            self,
            text="",
            font=("Arial", 13, "bold"),
            wraplength=790,
        )
        self.result.pack(pady=6)

        tk.Label(self, text="Key / Feature Flow", font=("Arial", 11, "bold")).pack(pady=(8, 2))
        self.flow_text = tk.Text(self, height=15, width=104)
        self.flow_text.pack(pady=5)

        tk.Label(
            self,
            text="Measured Size, Actual Time, and Estimated Network Time",
            font=("Arial", 11, "bold"),
        ).pack(pady=(8, 2))
        self.timing_text = tk.Text(self, height=18, width=104)
        self.timing_text.pack(pady=5)

        # 실행 시 기본 사용자의 키가 존재하면 자동 로드한다.
        self.after(100, self._try_initial_key_load)

    # ── 스레드 및 UI 헬퍼 ────────────────────────────────────────────────────
    def _run_in_thread(self, task) -> None:
        if self._busy:
            self._set_status("Status: 이전 작업이 진행 중입니다.")
            return
        self._busy = True
        threading.Thread(target=self._task_wrapper, args=(task,), daemon=True).start()

    def _task_wrapper(self, task) -> None:
        try:
            task()
        except Exception as exc:
            self.after(0, messagebox.showerror, "Error", str(exc))
            self._set_status(f"Status: failed — {exc}")
        finally:
            self._busy = False

    def _set_status(self, text: str) -> None:
        self.after(0, self.status.config, {"text": text})

    def _clear_outputs(self) -> None:
        self.after(0, self.flow_text.delete, "1.0", "end")
        self.after(0, self.timing_text.delete, "1.0", "end")

    def _log_flow(self, text: str) -> None:
        print(f"[flow] {text}")
        self.after(0, self.flow_text.insert, "end", text + "\n")

    def _log_timing(self, text: str) -> None:
        print(f"[timing] {text}")
        self.after(0, self.timing_text.insert, "end", text + "\n")

    # ── 입력 처리 ─────────────────────────────────────────────────────────────
    def read_values(self) -> list[float]:
        values: list[float] = []
        for label, key, _ in FEATURES:
            text = self.entries[key].get().strip()
            if not text:
                raise ValueError(f"{label} 값이 비어 있습니다.")
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError(f"{label} 값은 숫자여야 합니다: {text}") from exc
        self._validate_values(values)
        return values

    @staticmethod
    def _validate_values(values: list[float]) -> None:
        spo2, avg_hr, slptime, slp_eff, timest34p, age, sex, bmi = values
        checks = [
            (0.0 <= spo2 <= 100.0, "SpO2는 0~100 범위여야 합니다."),
            (20.0 <= avg_hr <= 250.0, "평균 심박수는 20~250 범위여야 합니다."),
            (0.0 < slptime <= 1440.0, "총 수면시간은 0 초과 1440분 이하여야 합니다."),
            (0.0 <= slp_eff <= 1.0, "수면 효율은 0~1 범위여야 합니다."),
            (0.0 <= timest34p <= 1.0, "깊은 수면 비율은 0~1 범위여야 합니다."),
            (0.0 <= age <= 120.0, "나이는 0~120 범위여야 합니다."),
            (sex in (0.0, 1.0), "성별 값은 0 또는 1이어야 합니다."),
            (5.0 <= bmi <= 100.0, "BMI는 5~100 범위여야 합니다."),
        ]
        for valid, message in checks:
            if not valid:
                raise ValueError(message)

    def load_watch_data(self) -> None:
        if extract_features is None:
            messagebox.showerror("Watch data error", "feature_processor.py를 찾을 수 없습니다.")
            return
        path = filedialog.askopenfilename(
            title="Select watch data file",
            filetypes=[("Watch data files", "*.csv *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            feats = extract_features(path)
            mapping = {
                "spo2": "spo2",
                "avg_hr": "avg_hr",
                "slptime": "total_sleep_time",
                "slp_eff": "sleep_efficiency",
                "timest34p": "deep_sleep_ratio",
                "age": "age",
                "sex": "sex",
                "bmi": "bmi",
            }
            for gui_key, feat_key in mapping.items():
                if feat_key in feats:
                    self.entries[gui_key].delete(0, "end")
                    self.entries[gui_key].insert(0, str(feats[feat_key]))
            self.status.config(text=f"Status: loaded watch data from {path}")
        except Exception as exc:
            messagebox.showerror("Watch data error", str(exc))

    # ── Phase 1. 사용자별 키 생성/저장/로드 ───────────────────────────────────
    def _try_initial_key_load(self) -> None:
        if ts is None:
            return
        user_id = self.user_id_entry.get().strip()
        try:
            paths = _key_paths(user_id)
        except ValueError:
            return
        if paths.private_context.exists() and paths.public_context.exists():
            self._run_in_thread(lambda: self._load_user_keys(user_id))

    def _create_user_keys_task(self) -> None:
        if ts is None:
            raise RuntimeError(f"TenSEAL not available: {IMPORT_ERROR}")

        user_id = self.user_id_entry.get().strip()
        paths = _key_paths(user_id)
        self._set_status(f"Status: {user_id} 사용자 CKKS 키 생성 중...")

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60],
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        context.generate_relin_keys()

        # 클라이언트 전용: secret/public/evaluation key가 모두 포함된다.
        private_bytes = context.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=True,
            save_relin_keys=True,
        )

        # 병원 배포용: Secret Key를 제거한다.
        public_context = context.copy()
        public_context.make_context_public()
        public_bytes = public_context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True,
        )
        context_id = _context_id(public_bytes)

        _atomic_write(paths.private_context, private_bytes)
        _atomic_write(paths.public_context, public_bytes)
        metadata = {
            "user_id": user_id,
            "context_id": context_id,
            "created_at_epoch": time.time(),
            "poly_modulus_degree": 16384,
            "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 40, 60],
            "global_scale": "2^40",
            "private_context_bytes": len(private_bytes),
            "public_context_bytes": len(public_bytes),
        }
        _atomic_write(paths.metadata, json.dumps(metadata, indent=2).encode("utf-8"))

        self.context = context
        self.public_context_bytes = public_bytes
        self.current_user_id = user_id
        self.current_context_id = context_id
        self.synced_context_ids.clear()

        self._clear_outputs()
        self._log_flow(f"사용자: {user_id}")
        self._log_flow(f"Context ID: {context_id}")
        self._log_flow(f"Private context 저장: {paths.private_context}")
        self._log_flow(f"Public context 저장:  {paths.public_context}")
        self._log_flow("Secret Key는 private_context.bin에만 있으며 병원으로 전송하지 않음")
        self._log_timing(f"Private context 크기: {_format_bytes(len(private_bytes))}")
        self._log_timing(f"Public context 크기:  {_format_bytes(len(public_bytes))}")
        self._set_status(f"Status: {user_id} 사용자 키 생성 및 파일 저장 완료.")

    def _load_user_keys(self, user_id: str) -> None:
        if ts is None:
            raise RuntimeError(f"TenSEAL not available: {IMPORT_ERROR}")

        paths = _key_paths(user_id)
        if not paths.private_context.exists() or not paths.public_context.exists():
            raise FileNotFoundError(
                f"{user_id} 사용자의 저장된 키가 없습니다. 먼저 'Create/Replace User Keys'를 실행하세요."
            )

        self._set_status(f"Status: {user_id} 사용자 저장 키 로드 중...")
        private_bytes = paths.private_context.read_bytes()
        public_bytes = paths.public_context.read_bytes()

        loaded_context = ts.context_from(private_bytes)
        if not loaded_context.has_secret_key():
            raise RuntimeError("저장된 private context에 Secret Key가 없습니다.")

        self.context = loaded_context
        self.public_context_bytes = public_bytes
        self.current_user_id = user_id
        self.current_context_id = _context_id(public_bytes)

        self._log_flow(f"저장된 키 로드 완료: user={user_id}, context_id={self.current_context_id}")
        self._log_flow(f"Secret context 경로: {paths.private_context}")
        self._log_flow("병원 전송 대상은 public_context.bin뿐임")
        self._set_status(f"Status: {user_id} 사용자 저장 키 로드 완료.")

    def _ensure_user_context_loaded(self) -> None:
        user_id = self.user_id_entry.get().strip()
        if self.context is not None and self.current_user_id == user_id:
            return
        self._load_user_keys(user_id)

    # ── Phase 2~6. 전체 추론 자동 실행 ───────────────────────────────────────
    def _automatic_inference_task(self) -> None:
        if ts is None:
            raise RuntimeError(f"TenSEAL not available: {IMPORT_ERROR}")

        self._clear_outputs()
        total_start = time.perf_counter()

        # Phase 1: 저장된 사용자별 키/context 로드
        self._set_status("Status: 1/6 저장된 사용자 키 로드 중...")
        self._ensure_user_context_loaded()
        assert self.context is not None
        assert self.public_context_bytes is not None
        assert self.current_context_id is not None

        # Phase 2: GUI feature 확인 및 정규화
        self._set_status("Status: 2/6 GUI 입력 확인 및 정규화 중...")
        raw_values = self.read_values()
        normalized = normalize(raw_values)
        self._log_flow("[1] GUI 입력 feature 확인")
        for (label, _, _), raw, norm in zip(FEATURES, raw_values, normalized):
            self._log_flow(f"  {label}: raw={raw:.6g} -> normalized={norm:.6g}")

        # Phase 3: 병원별 feature만 암호화 → serialize → Base64 → JSON 구성
        self._set_status("Status: 3/6 병원별 feature 암호화 및 JSON 구성 중...")
        request_payloads: dict[int, dict[str, Any]] = {}
        ciphertext_sizes: dict[int, int] = {}
        base64_sizes: dict[int, int] = {}
        json_sizes: dict[int, int] = {}

        self._log_flow("[2] 병원별 feature 분할 → CKKS 암호화 → 직렬화 → Base64 → JSON")
        for hospital_id, indices in enumerate(FEATURE_GROUPS):
            feature_slice = [normalized[index] for index in indices]
            padded_len = _next_pow2(len(feature_slice))
            padded = feature_slice + [0.0] * (padded_len - len(feature_slice))

            encrypted = ts.ckks_vector(self.context, padded)
            ciphertext_bytes = encrypted.serialize()
            ciphertext_b64 = base64.b64encode(ciphertext_bytes).decode("ascii")
            payload = {
                "user_id": self.current_user_id,
                "context_id": self.current_context_id,
                "hospital_id": hospital_id,
                "feature_indices": indices,
                "enc_xi_b64": ciphertext_b64,
            }

            request_payloads[hospital_id] = payload
            ciphertext_sizes[hospital_id] = len(ciphertext_bytes)
            base64_sizes[hospital_id] = len(ciphertext_b64.encode("ascii"))
            json_sizes[hospital_id] = _json_size(payload)

            self._log_flow(
                f"  Hospital {hospital_id}: indices={indices}, plaintext={feature_slice}, "
                f"padded_len={padded_len}, ciphertext={_format_bytes(len(ciphertext_bytes))}"
            )

        # Phase 4: public context는 최초/강제 시에만 병원별 직접 동기화
        self._set_status("Status: 4/6 병원 public context 상태 확인 중...")
        context_upload_results = self._sync_contexts_if_needed()

        # Phase 5: 세 병원에 병렬로 직접 전송. 병원 간 사용자 데이터 전달 없음.
        self._set_status("Status: 5/6 병원 A/B/C에 직접 병렬 추론 요청 중...")
        inference_start = time.perf_counter()
        hospital_results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(HOSPITAL_URLS)) as executor:
            future_map = {
                executor.submit(self._call_hospital, hospital_id, request_payloads[hospital_id]): hospital_id
                for hospital_id in range(len(HOSPITAL_URLS))
            }
            for future in as_completed(future_map):
                hospital_id = future_map[future]
                hospital_results[hospital_id] = future.result()
        parallel_inference_ms = (time.perf_counter() - inference_start) * 1000.0

        # Phase 6: 응답 암호문을 클라이언트에서 합산하고 Secret Key로 복호화
        self._set_status("Status: 6/6 결과 취합, 합산 및 클라이언트 복호화 중...")
        encrypted_logit = None
        response_sizes: dict[int, int] = {}
        for hospital_id in sorted(hospital_results):
            result = hospital_results[hospital_id]
            share_b64 = result["body"]["enc_logit_share_b64"]
            response_sizes[hospital_id] = result["response_bytes"]
            share = ts.ckks_vector_from(self.context, base64.b64decode(share_b64))
            encrypted_logit = share if encrypted_logit is None else encrypted_logit + share

        if encrypted_logit is None:
            raise RuntimeError("병원에서 받은 암호화 logit share가 없습니다.")

        decrypt_start = time.perf_counter()
        logit = encrypted_logit.decrypt()[0]
        decrypt_ms = (time.perf_counter() - decrypt_start) * 1000.0
        probability = sigmoid(logit)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        self._report_measurements(
            context_upload_results=context_upload_results,
            ciphertext_sizes=ciphertext_sizes,
            base64_sizes=base64_sizes,
            json_sizes=json_sizes,
            hospital_results=hospital_results,
            response_sizes=response_sizes,
            parallel_inference_ms=parallel_inference_ms,
            decrypt_ms=decrypt_ms,
            total_ms=total_ms,
        )

        def update_ui() -> None:
            self.result.config(
                text=f"logit: {logit:.4f}   |   수면무호흡 위험도: {probability * 100:.2f}%"
            )
            self.status.config(
                text="Status: 키 재사용 → 병원별 직접 전송 → 결과 취합 → 로컬 복호화 완료. "
                     "입력값을 바꾼 뒤 다시 Run Secure Inference를 누를 수 있습니다."
            )

        self.after(0, update_ui)

    def _sync_contexts_if_needed(self) -> dict[int, dict[str, Any]]:
        assert self.public_context_bytes is not None
        assert self.current_context_id is not None

        force = bool(self.force_context_upload.get())
        results: dict[int, dict[str, Any]] = {}

        def upload(hospital_id: int) -> tuple[int, dict[str, Any]]:
            if not force and self.synced_context_ids.get(hospital_id) == self.current_context_id:
                return hospital_id, {
                    "uploaded": False,
                    "elapsed_ms": 0.0,
                    "request_bytes": 0,
                    "response_bytes": 0,
                }

            url = HOSPITAL_URLS[hospital_id]
            start = time.perf_counter()
            response = requests.post(
                f"{url}/upload_context",
                data=self.public_context_bytes,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-User-ID": self.current_user_id or "",
                    "X-Context-ID": self.current_context_id,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            self.synced_context_ids[hospital_id] = self.current_context_id
            return hospital_id, {
                "uploaded": True,
                "elapsed_ms": elapsed_ms,
                "request_bytes": len(self.public_context_bytes),
                "response_bytes": len(response.content),
            }

        with ThreadPoolExecutor(max_workers=len(HOSPITAL_URLS)) as executor:
            futures = [executor.submit(upload, hospital_id) for hospital_id in range(len(HOSPITAL_URLS))]
            for future in as_completed(futures):
                hospital_id, result = future.result()
                results[hospital_id] = result

        return results

    @staticmethod
    def _call_hospital(hospital_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        url = HOSPITAL_URLS[hospital_id]
        start = time.perf_counter()
        response = requests.post(
            f"{url}/compute_logit_share",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        body = response.json()
        if "enc_logit_share_b64" not in body:
            raise RuntimeError(f"Hospital {hospital_id} 응답에 enc_logit_share_b64가 없습니다.")
        return {
            "elapsed_ms": elapsed_ms,
            "body": body,
            "response_bytes": len(response.content),
        }

    def _report_measurements(
        self,
        *,
        context_upload_results: dict[int, dict[str, Any]],
        ciphertext_sizes: dict[int, int],
        base64_sizes: dict[int, int],
        json_sizes: dict[int, int],
        hospital_results: dict[int, dict[str, Any]],
        response_sizes: dict[int, int],
        parallel_inference_ms: float,
        decrypt_ms: float,
        total_ms: float,
    ) -> None:
        assert self.public_context_bytes is not None

        self._log_timing("=== 실제 측정 데이터 크기 ===")
        self._log_timing(f"Public context: {_format_bytes(len(self.public_context_bytes))}")
        for hospital_id in range(len(HOSPITAL_URLS)):
            self._log_timing(
                f"Hospital {hospital_id}: ciphertext(raw)={_format_bytes(ciphertext_sizes[hospital_id])}, "
                f"Base64={_format_bytes(base64_sizes[hospital_id])}, "
                f"JSON request={_format_bytes(json_sizes[hospital_id])}, "
                f"JSON response={_format_bytes(response_sizes[hospital_id])}"
            )

        self._log_timing("\n=== Docker/현재 환경 실제 왕복시간 ===")
        for hospital_id in range(len(HOSPITAL_URLS)):
            context_result = context_upload_results[hospital_id]
            if context_result["uploaded"]:
                self._log_timing(
                    f"Hospital {hospital_id} public context upload RTT: "
                    f"{context_result['elapsed_ms']:.1f} ms"
                )
            else:
                self._log_timing(f"Hospital {hospital_id} public context upload: 생략(cache 재사용)")
            self._log_timing(
                f"Hospital {hospital_id} inference RTT: "
                f"{hospital_results[hospital_id]['elapsed_ms']:.1f} ms"
            )
        self._log_timing(f"세 병원 병렬 inference wall time: {parallel_inference_ms:.1f} ms")
        self._log_timing(f"Client decrypt time: {decrypt_ms:.1f} ms")
        self._log_timing(f"전체 자동 실행 wall time: {total_ms:.1f} ms")

        self._log_timing("\n=== 가상 네트워크 예상 통신시간 ===")
        for scenario_name, scenario in NETWORK_SCENARIOS.items():
            bandwidth = scenario["bandwidth_mbps"]
            rtt = scenario["rtt_ms"]

            # 직접 병렬 전송이므로 wall-clock 예상값은 가장 느린 병원 요청을 기준으로 한다.
            infer_roundtrips = []
            context_roundtrips = []
            for hospital_id in range(len(HOSPITAL_URLS)):
                request_bytes = json_sizes[hospital_id]
                response_bytes = response_sizes[hospital_id]
                infer_ms = (
                    _estimated_one_way_ms(request_bytes + response_bytes, bandwidth) + rtt
                )
                infer_roundtrips.append(infer_ms)

                context_ms = (
                    _estimated_one_way_ms(len(self.public_context_bytes), bandwidth) + rtt
                )
                context_roundtrips.append(context_ms)

            cached_wall_ms = max(infer_roundtrips)
            first_wall_ms = max(context_roundtrips) + cached_wall_ms
            repeated_total_bytes = sum(json_sizes.values()) + sum(response_sizes.values())
            first_total_bytes = repeated_total_bytes + len(self.public_context_bytes) * len(HOSPITAL_URLS)

            self._log_timing(
                f"{scenario_name} ({bandwidth:g} Mbps, RTT {rtt:g} ms): "
                f"최초(context 포함)≈{first_wall_ms:.1f} ms, "
                f"재추론(context 생략)≈{cached_wall_ms:.1f} ms, "
                f"총 전송량 최초={_format_bytes(first_total_bytes)}, "
                f"재추론={_format_bytes(repeated_total_bytes)}"
            )

        self._log_timing(
            "\n계산식: 예상 통신시간 = (요청+응답 byte × 8 / bandwidth) + RTT. "
            "세 병원 요청은 병렬이므로 합계가 아니라 최대값을 wall time으로 사용. "
            "서버의 HE 연산시간은 별도이므로 위 예상치는 순수 네트워크 시간이다."
        )


if __name__ == "__main__":
    app = SecureHealthClient()
    app.mainloop()
