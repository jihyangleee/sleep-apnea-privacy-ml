"""NSRR에서 SHHS1 EDF를 한 개씩 받아 SaO2 채널만 남기고 원본은 지운다.

SHHS1 EDF 전체는 ~230GB라 디스크에 다 둘 수 없다. 파일마다
  다운로드(임시) → SaO2 채널 추출 → shhs/sao2/<이름>.npz 저장 → EDF 삭제
를 반복하므로 디스크는 동시 다운로드 수 × ~40MB + 결과물(수 GB)만 쓴다.
이미 .npz가 있는 파일은 건너뛰므로 중간에 멈춰도 다시 실행하면 이어서 받는다.

토큰은 https://sleepdata.org/token 에서 받아 환경변수로 넘긴다 (코드·로그에 남기지 않는다).
  PowerShell:  $env:NSRR_TOKEN = "<토큰>"; python download_shhs_sao2.py --limit 3
  bash:        NSRR_TOKEN=<토큰> python download_shhs_sao2.py --limit 3

출력 .npz:  sao2 (float32, 물리 단위 %), fs (Hz), label, phys_dim
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ssl

import numpy as np
import requests
from requests.adapters import HTTPAdapter

WEBSITE = "https://sleepdata.org"
DATASET = "shhs"
REMOTE_DIR = "polysomnography/edfs/shhs1"
GEM_TAG = "nsrr-gem-v8-0-0"  # nsrr gem 8.0.0과 같은 다운로드 경로
OUT_DIR = os.path.join("shhs", "sao2")
TMP_DIR = os.path.join("shhs", "_edf_tmp")
SAO2_LABELS = ("SAO2", "SPO2", "SA02")


def load_env_token(path=".env"):
    """.env에서 토큰을 읽는다. 'NSRR_TOKEN=값', 'token=값', 'token : 값' 형식을 모두 받는다."""
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for sep in ("=", ":"):
                if sep in line:
                    key, val = line.split(sep, 1)
                    if key.strip().lower() in ("nsrr_token", "token"):
                        return val.strip().strip('"').strip("'")
                    break
    return ""


class _SystemCertAdapter(HTTPAdapter):
    """certifi 묶음 대신 OS 인증서 저장소로 검증한다 (certifi로는 sleepdata.org 체인 검증이 실패한다)."""

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = ssl.create_default_context()
        return super().init_poolmanager(*args, **kwargs)


def make_session(workers):
    s = requests.Session()
    s.mount("https://", _SystemCertAdapter(pool_connections=workers, pool_maxsize=workers))
    return s


LIST_CACHE = os.path.join("shhs", "sao2_filelist.json")


def list_remote_files_cached(token, session, retries=12):
    """파일 목록을 한 번 받으면 캐시한다. NSRR 서버가 502/타임아웃을 자주 내므로 간격을 늘려 재시도한다."""
    if os.path.exists(LIST_CACHE):
        with open(LIST_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    for attempt in range(retries):
        try:
            files = list_remote_files(token, session)
            with open(LIST_CACHE, "w", encoding="utf-8") as fh:
                json.dump(files, fh)
            return files
        except Exception as e:
            wait = min(30 * 2 ** attempt, 600)
            print(f"[list] 실패 ({attempt + 1}/{retries}): {str(e).replace(token, '<token>')[:150]} → {wait}s 후 재시도", flush=True)
            time.sleep(wait)
    raise RuntimeError("파일 목록을 받지 못했다")


def list_remote_files(token, session):
    r = session.get(f"{WEBSITE}/api/v1/datasets/{DATASET}/files.json",
                     params={"auth_token": token, "path": REMOTE_DIR}, timeout=60)
    r.raise_for_status()
    files = [f for f in r.json() if f.get("is_file") and f["file_name"].lower().endswith(".edf")]
    return sorted(files, key=lambda f: f["file_name"])


def read_edf_channel(path, wanted=SAO2_LABELS):
    """EDF에서 한 채널만 읽는다 (외부 라이브러리 없이 EDF 스펙대로 파싱)."""
    sig = read_edf_channels(path, {"x": wanted}, required=("x",))["x"]
    return sig["data"], sig["fs"], sig["label"], sig["phys_dim"]


def read_edf_channels(path, wanted, required=()):
    """{key: (label 후보, ...)} → {key: {data, fs, label, phys_dim}}. 없는 채널은 결과에서 빠진다."""
    with open(path, "rb") as fh:
        head = fh.read(256)
        n_records = int(head[236:244].decode().strip())
        rec_dur = float(head[244:252].decode().strip())
        ns = int(head[252:256].decode().strip())
        sig = fh.read(ns * 256)

        def field(offset, width):
            return [sig[offset * ns + i * width: offset * ns + (i + 1) * width].decode("latin-1").strip()
                    for i in range(ns)]

        labels = field(0, 16)
        phys_dim = field(16 + 80, 8)
        pmin = [float(x) for x in field(16 + 80 + 8, 8)]
        pmax = [float(x) for x in field(16 + 80 + 16, 8)]
        dmin = [float(x) for x in field(16 + 80 + 24, 8)]
        dmax = [float(x) for x in field(16 + 80 + 32, 8)]
        spr = [int(x) for x in field(16 + 80 + 40 + 80, 8)]

        norm = [l.upper().replace(" ", "") for l in labels]
        found = {}
        for key, cands in wanted.items():
            idx = next((i for i, l in enumerate(norm) if any(w in l for w in cands)), None)
            if idx is not None:
                found[key] = idx
            elif key in required:
                raise ValueError(f"{key} 채널 없음: {labels}")

        data = np.frombuffer(fh.read(), dtype="<i2")

    rec_len = sum(spr)
    n_records = min(n_records, len(data) // rec_len) if n_records > 0 else len(data) // rec_len
    data = data[: n_records * rec_len].reshape(n_records, rec_len)
    out = {}
    for key, idx in found.items():
        start = sum(spr[:idx])
        raw = data[:, start:start + spr[idx]].reshape(-1).astype(np.float64)
        gain = (pmax[idx] - pmin[idx]) / (dmax[idx] - dmin[idx])
        phys = (raw - dmin[idx]) * gain + pmin[idx]
        out[key] = {"data": phys.astype(np.float32), "fs": spr[idx] / rec_dur,
                    "label": labels[idx], "phys_dim": phys_dim[idx]}
    return out


# SaO2와 같은 1Hz로 기록된 산소측정기 보조 채널. 워치의 1분 심박과 맞춰 심박 변동 피처를 만들고,
# OX stat(산소측정기 상태)으로 SpO2 잡음 구간을 거른다.
CHANNELS = {"sao2": SAO2_LABELS, "hr": ("H.R.",), "ox_stat": ("OXSTAT",)}


def has_all_channels(npz_path):
    # OX stat 채널이 없는 EDF도 있으므로 완료 판정은 심박(hr) 유무로 한다 (없으면 재시작마다 다시 받게 됨)
    with np.load(npz_path) as d:
        return "hr" in d.files


def is_done(f, refetch):
    out = os.path.join(OUT_DIR, f["file_name"][:-4] + ".npz")
    return os.path.exists(out) and (not refetch or has_all_channels(out))


def process(f, token, session, refetch=False, extra_channels=False):
    name = f["file_name"]
    out = os.path.join(OUT_DIR, name[:-4] + ".npz")
    if is_done(f, refetch):
        return name, 0, "skip"
    tmp = os.path.join(TMP_DIR, name)
    url = f"{WEBSITE}/datasets/{DATASET}/files/a/{token}/m/{GEM_TAG}/{f['full_path']}"
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            size = os.path.getsize(tmp)
            if f.get("file_size") and size != int(f["file_size"]):
                raise IOError(f"크기 불일치 {size} != {f['file_size']}")
            wanted = CHANNELS if extra_channels else {"sao2": SAO2_LABELS}
            ch = read_edf_channels(tmp, wanted, required=("sao2",))
            s = ch["sao2"]
            extra = {}
            for k in ("hr", "ox_stat"):
                if k in ch:
                    extra[k], extra[f"{k}_fs"] = ch[k]["data"], ch[k]["fs"]
            np.savez_compressed(out, sao2=s["data"], fs=s["fs"], label=s["label"], phys_dim=s["phys_dim"], **extra)
            return name, size, f"ok fs={s['fs']:g}Hz n={len(s['data'])} +{','.join(k for k in ('hr', 'ox_stat') if k in ch) or 'none'}"
        except Exception as e:  # 네트워크 끊김 등은 재시도
            err = e
            time.sleep(5 * (attempt + 1))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return name, 0, f"FAIL {type(err).__name__}: {str(err).replace(token, '<token>')[:200]}"


def keep_awake():
    """Windows가 실행 중에 유휴 절전으로 들어가지 않게 한다 (프로세스가 끝나면 자동 해제).
    덮개를 닫거나 직접 절전을 누르는 것은 막지 못한다."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4, help="동시 다운로드 수")
    p.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (시험용)")
    p.add_argument("--extra-channels", action="store_true",
                   help="SaO2 외에 hr/ox_stat 채널도 저장한다 (기본: SaO2만)")
    p.add_argument("--refetch-missing", action="store_true",
                   help="--extra-channels와 함께: SaO2만 저장된 이전 .npz도 다시 받아 hr/ox_stat 채널을 채운다")
    a = p.parse_args()

    token = os.environ.get("NSRR_TOKEN", "").strip() or load_env_token()
    if not token:
        sys.exit("NSRR_TOKEN 환경변수나 .env의 token 항목이 없다. https://sleepdata.org/token 에서 발급받아 설정한다.")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    keep_awake()
    session = make_session(a.workers)
    try:
        files = list_remote_files_cached(token, session)
    except Exception as e:  # 오류 메시지의 URL에 토큰이 들어가므로 가린다
        sys.exit(f"파일 목록 요청 실패: {str(e).replace(token, '<token>')}")
    total_bytes = sum(int(f.get("file_size") or 0) for f in files)
    print(f"[list] {len(files)}개 EDF, 총 {total_bytes / 1e9:.1f} GB", flush=True)
    todo = [f for f in files if not is_done(f, a.refetch_missing)]
    print(f"[status] 이미 완료 {len(files) - len(todo)}개", flush=True)
    if a.limit:
        todo = todo[: a.limit]
    todo_bytes = sum(int(f.get("file_size") or 0) for f in todo)
    print(f"[todo] 이번 실행 {len(todo)}개, {todo_bytes / 1e9:.1f} GB", flush=True)

    lock, done_bytes, done, fails = threading.Lock(), 0, 0, []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, f, token, session, a.refetch_missing, a.extra_channels) for f in todo]
        for fut in as_completed(futs):
            name, size, msg = fut.result()
            with lock:
                done += 1
                done_bytes += size
                if msg.startswith("FAIL"):
                    fails.append(name)
                el = time.time() - t0
                rate = done_bytes / el if el > 0 else 0
                eta_h = (todo_bytes - done_bytes) / rate / 3600 if rate > 0 else float("nan")
                print(f"[{done}/{len(todo)}] {name} {msg} | {rate / 1e6:.1f} MB/s | 남은 시간 ~{eta_h:.1f}h", flush=True)

    print(f"[done] {done - len(fails)}개 성공, {len(fails)}개 실패 ({(time.time() - t0) / 3600:.2f}h)")
    if fails:
        print("  실패:", " ".join(fails))


if __name__ == "__main__":
    main()
