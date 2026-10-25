"""SpO2 1분 단위 피처 — SHHS(EDF 1Hz)와 갤럭시 워치(1분 binning)에 같은 함수를 쓴다.

워치가 주는 SpO2는 1분마다 (평균, 최소, 최대)뿐이다. 학습·추론의 피처 정의를 맞추려고
SHHS의 1Hz SaO2도 같은 1분 형식으로 묶은 뒤, 아래 features_from_bins() 하나로 둘 다 계산한다.

XML의 "SpO2 desaturation" 주석은 호흡 사건(기류 신호로 판정)에 묶인 저하만 담고 있어
라벨(AHI) 누출이 생기므로 쓰지 않는다. 여기서는 SpO2 신호만으로 저하를 찾는다.

피처 (모두 유효 분 기준):
    avg_spo2        분당 평균의 평균 (%)
    min_spo2        분당 최소의 하위 1% 값 (%) — 순수 최소는 순간 잡음에 끌려 내려간다
    pct_lt90        분당 최소 < 90% 인 분의 비율 (%) — 분당 평균 기준은 짧은 저하를 놓친다
    odi_1min        저하 사건 수 / 유효 시간 (회/h)
    desat_drop_1min 사건별 (기준선 - 구간 최저)의 평균 (%p)
    desat_min_frac  저하 사건에 걸친 분의 비율 (%)

저하 사건 규칙 (1분 단위):
    기준선 = 직전 BASELINE_MIN 개 유효 분의 평균 중 최대값
    분 최소 <= 기준선 - DROP_THR 이면 사건, 이어지는 분은 같은 기준선으로 한 사건에 합친다.
"""

import glob
import json
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np

DROP_THR = 3.0       # %p, 임상 ODI3% 기준
BASELINE_MIN = 3     # 기준선 계산에 쓰는 직전 유효 분 수
VALID_LO, VALID_HI = 50.0, 100.0  # 이 범위 밖 샘플은 센서 이탈/잡음으로 버린다
MIN_VALID_SEC = 30   # 1분 중 유효 샘플이 이보다 적으면 그 분은 무효
MIN_PCTL = 1         # min_spo2: 분당 최소값의 이 백분위수 (SHHS minsat과의 일치도 기준으로 선택)

FEATURES = ["avg_spo2", "min_spo2", "pct_lt90", "odi_1min", "desat_drop_1min", "desat_min_frac"]


# ── 1분 binning ──────────────────────────────────────────────────────────────

def bins_from_1hz(sao2, fs=1.0, lo=VALID_LO, hi=VALID_HI):
    """1Hz 신호 → (mean, min, max, valid) 1분 배열. 워치 binning과 같은 형식. 심박도 lo/hi만 바꿔 쓴다."""
    per_min = int(round(60 * fs))
    n = len(sao2) // per_min
    x = np.asarray(sao2[: n * per_min], dtype=float).reshape(n, per_min)
    ok = (x >= lo) & (x <= hi)
    cnt = ok.sum(axis=1)
    valid = cnt >= MIN_VALID_SEC * fs
    xm = np.where(ok, x, np.nan)
    with warnings.catch_warnings():  # 유효 샘플이 없는 분은 NaN이 되고 valid=False로 걸러진다
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(xm, axis=1)
        mn = np.nanmin(xm, axis=1)
        mx = np.nanmax(xm, axis=1)
    return mean, mn, mx, valid


def bins_from_samsung(path):
    """Samsung Health oxygen_saturation binning.json → (mean, min, max, valid, start_ms)."""
    with open(path, encoding="utf-8") as fh:
        rows = sorted(json.load(fh), key=lambda r: r["start_time"])
    if not rows:
        return (np.array([]),) * 4 + (None,)
    t0 = rows[0]["start_time"]
    n = int((rows[-1]["start_time"] - t0) // 60000) + 1
    mean, mn, mx = (np.full(n, np.nan) for _ in range(3))
    for r in rows:  # 빠진 분은 NaN(무효)으로 남겨 시간축을 SHHS와 같게 유지
        i = int((r["start_time"] - t0) // 60000)
        mean[i], mn[i], mx[i] = r.get("spo2", np.nan), r.get("spo2_min", np.nan), r.get("spo2_max", np.nan)
    valid = (mean >= VALID_LO) & (mean <= VALID_HI) & (mn >= VALID_LO)
    return mean, mn, mx, valid, t0


# ── 피처 계산 (SHHS·워치 공통) ────────────────────────────────────────────────

def desat_events(mean, mn, valid, thr=DROP_THR, base_n=BASELINE_MIN):
    """[(시작 분, 끝 분, 저하 폭), ...]"""
    events, hist, i, n = [], [], 0, len(mean)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        if len(hist) >= base_n:
            base = max(hist[-base_n:])
            if mn[i] <= base - thr:
                j, low = i, mn[i]
                while j + 1 < n and valid[j + 1] and mn[j + 1] <= base - thr:
                    j += 1
                    low = min(low, mn[j])
                events.append((i, j, base - low))
                hist.extend(mean[i:j + 1][valid[i:j + 1]])
                i = j + 1
                continue
        hist.append(mean[i])
        i += 1
    return events


def features_from_bins(mean, mn, mx, valid, thr=DROP_THR):
    n_valid = int(valid.sum())
    if n_valid < 60:  # 유효 1시간 미만이면 계산하지 않는다
        return None
    hours = n_valid / 60
    ev = desat_events(mean, mn, valid, thr)
    ev_min = sum(e[1] - e[0] + 1 for e in ev)
    return {
        "avg_spo2": float(np.mean(mean[valid])),
        "min_spo2": float(np.percentile(mn[valid], MIN_PCTL)),
        "pct_lt90": 100.0 * float(np.mean(mn[valid] < 90)),
        "odi_1min": len(ev) / hours,
        "desat_drop_1min": float(np.mean([e[2] for e in ev])) if ev else 0.0,
        "desat_min_frac": 100.0 * ev_min / n_valid,
        "valid_hours": hours,
    }


# ── 심박 1분 피처 (주기적 심박 변동, CVHR을 1분 해상도에 맞춘 근사) ───────────────
# 무호흡이 끝나며 각성하면 심박이 급증한다. 박동 단위가 아니라 1분 (평균, 최소, 최대)만 있으므로
# "분 최대가 직전 기준선보다 HR_SURGE_BPM 이상 높은 분"을 급증으로 보고, 이어지는 분은 한 번으로 센다.
# 규칙은 성능을 보기 전에 고정했다 (결과를 보고 바꾸면 성능이 과대평가된다).

HR_LO, HR_HI = 30.0, 200.0
HR_SURGE_BPM = 6.0     # 기준선 대비 급증 폭
HR_BASELINE_MIN = 5    # 기준선 = 직전 5개 유효 분 평균의 중앙값
HR_FEATURES = ["hr_surge_idx", "hr_surge_frac", "hr_range_mean", "desat_hr_coupled"]


def hr_surges(mean, mx, valid, thr=HR_SURGE_BPM, base_n=HR_BASELINE_MIN):
    """[(시작 분, 끝 분), ...]"""
    events, hist, i, n = [], [], 0, len(mean)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        if len(hist) >= base_n:
            base = float(np.median(hist[-base_n:]))
            if mx[i] >= base + thr:
                j = i
                while j + 1 < n and valid[j + 1] and mx[j + 1] >= base + thr:
                    j += 1
                events.append((i, j))
                hist.extend(mean[i:j + 1][valid[i:j + 1]])
                i = j + 1
                continue
        hist.append(mean[i])
        i += 1
    return events


def hr_features_from_bins(mean, mn, mx, valid, desat_ev=None):
    """심박 1분 bin → 피처. desat_ev: 같은 시간축의 SpO2 저하 사건 (desat_events 결과)."""
    n_valid = int(valid.sum())
    if n_valid < 60:
        return None
    ev = hr_surges(mean, mx, valid)
    surge_min = np.zeros(len(mean), bool)
    for a, b in ev:
        surge_min[a:b + 1] = True
    f = {
        "hr_surge_idx": len(ev) / (n_valid / 60),                     # 급증 횟수 / 시간
        "hr_surge_frac": 100.0 * surge_min[valid].mean(),             # 급증 분 비율 (%)
        "hr_range_mean": float(np.mean((mx - mn)[valid])),            # 분 안 최대-최소 평균 (bpm)
    }
    if desat_ev is not None:
        # SpO2 저하 사건 중 같은 분~2분 뒤에 심박 급증이 있었던 비율 (%)
        hit = [surge_min[a:min(b + 3, len(surge_min))].any() for a, b, _ in desat_ev]
        f["desat_hr_coupled"] = 100.0 * float(np.mean(hit)) if hit else 0.0
    return f


def shhs_hr_features(npz_path, xml_dir):
    """SaO2와 H.R. 채널이 모두 있는 SHHS npz → SpO2 피처 + 심박 피처 (같은 수면 구간)."""
    d = np.load(npz_path)
    if "hr" not in d.files:
        return None
    name = os.path.basename(npz_path)[:-4]
    sao2, fs, hr, hfs = d["sao2"], float(d["fs"]), d["hr"], float(d["hr_fs"])
    win = shhs_sleep_window(os.path.join(xml_dir, f"{name}-nsrr.xml"))
    if win:
        sao2 = sao2[int(win[0] * fs): int(win[1] * fs)]
        hr = hr[int(win[0] * hfs): int(win[1] * hfs)]
    sb = bins_from_1hz(sao2, fs)
    hb = bins_from_1hz(hr, hfs, HR_LO, HR_HI)
    n = min(len(sb[0]), len(hb[0]))
    sb = tuple(x[:n] for x in sb)
    hb = tuple(x[:n] for x in hb)
    f = features_from_bins(*sb)
    h = hr_features_from_bins(*hb, desat_ev=desat_events(sb[0], sb[1], sb[3]))
    if f is None or h is None:
        return None
    f.update(h)
    f["nsrrid"] = int(name.split("-")[1])
    return f


# ── SHHS: 수면 구간 자르기 (XML 수면 단계 주석 — 워치도 수면 단계를 주므로 누출 아님) ──

def shhs_sleep_window(xml_path):
    """첫 수면 epoch 시작 ~ 마지막 수면 epoch 끝 (초). 수면 단계가 없으면 None."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return None
    sleep = []
    for e in root.iter("ScoredEvent"):
        if (e.findtext("EventType") or "").startswith("Stages") and not (e.findtext("EventConcept") or "").startswith("Wake"):
            s = float(e.findtext("Start"))
            sleep.append((s, s + float(e.findtext("Duration"))))
    if not sleep:
        return None
    return min(s for s, _ in sleep), max(e for _, e in sleep)


def shhs_features(npz_path, xml_dir, thr=DROP_THR):
    d = np.load(npz_path)
    sao2, fs = d["sao2"], float(d["fs"])
    name = os.path.basename(npz_path)[:-4]
    win = shhs_sleep_window(os.path.join(xml_dir, f"{name}-nsrr.xml"))
    if win:
        sao2 = sao2[int(win[0] * fs): int(win[1] * fs)]
    f = features_from_bins(*bins_from_1hz(sao2, fs), thr=thr)
    if f is not None:
        f["nsrrid"] = int(name.split("-")[1])
        f["sleep_cropped"] = win is not None
    return f


if __name__ == "__main__":
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(description="shhs/sao2/*.npz → 1분 SpO2 피처 CSV")
    p.add_argument("--sao2-dir", default="shhs/sao2")
    p.add_argument("--xml-dir", default="shhs/polysomnography/annotations-events-nsrr/shhs1")
    p.add_argument("--out", default="shhs/datasets/oximetry_1min_features.csv")
    p.add_argument("--thr", type=float, default=DROP_THR)
    a = p.parse_args()

    rows = [f for f in (shhs_features(x, a.xml_dir, a.thr) for x in sorted(glob.glob(os.path.join(a.sao2_dir, "*.npz")))) if f]
    df = pd.DataFrame(rows).set_index("nsrrid")
    df.to_csv(a.out)
    print(f"[oximetry] {len(df)}명 → {a.out}")
    print(df[FEATURES + ["valid_hours"]].describe().round(2).to_string())
