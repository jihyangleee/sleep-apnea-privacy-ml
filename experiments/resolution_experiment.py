"""1초 해상도 SpO2·심박 피처가 현재 16개(1분 해상도) 모델보다 나은지 평문 로지스틱 회귀로 비교한다.

워치가 초 단위 SpO2/심박을 줄 수 있다면 얻을 수 있는 성능을 미리 보는 실험이다. 배포 모델과 무관하다.
같은 shhs/sao2/*.npz(EDF에서 뽑은 1Hz SaO2, 일부는 H.R.)와 같은 수면 구간(XML 수면 단계)을 쓰고,
1분으로 묶는 단계만 빼고 1초 신호에서 바로 계산한다.

피처 규칙은 결과를 보기 전에 고정했다 (결과를 보고 바꾸면 성능이 과대평가된다):
  SpO2 1초 저하 사건: 기준선 = 직전 SPO2_BASE_SEC초 유효 샘플의 최대값.
                     값 <= 기준선 - 3%p 인 상태가 SPO2_MIN_SEC초 이상 이어지면 사건 1개 (임상 ODI3%).
  심박 1초 급증:     기준선 = 직전 HR_BASE_SEC초 유효 샘플의 중앙값.
                     값 >= 기준선 + 6bpm 인 상태가 HR_MIN_SEC초 이상 이어지면 급증 1개 (CVHR 근사).

비교 (같은 5겹 KFold, shuffle, seed 42, 5회 반복 / 표준화는 fold마다 학습 행으로만):
  SpO2: 전체 코호트(mpc/hospital_data, 5,355명)에서 16개 vs 1분 SpO2 5개를 1초 버전으로 바꾼 16개
  심박: H.R. 채널이 있는 사람만 16개 vs 16개 + 심박 1분 피처 vs 16개 + 심박 1초 피처

사용법 (프로젝트 루트에서): python experiments/resolution_experiment.py
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from oximetry_features import (
    DROP_THR, HR_FEATURES, HR_HI, HR_LO, VALID_HI, VALID_LO,
    bins_from_1hz, desat_events, hr_features_from_bins, shhs_sleep_window,
)

SAO2_DIR = "shhs/sao2"
XML_DIR = "shhs/polysomnography/annotations-events-nsrr/shhs1"
DATA_DIR = "mpc/hospital_data"
OUT_CSV = "shhs/datasets/resolution_1s_features.csv"
LABEL = "ahi_a0h3a"

SPO2_BASE_SEC = 120
SPO2_MIN_SEC = 10
HR_BASE_SEC = 60
HR_SURGE_BPM = 6.0
HR_MIN_SEC = 5
COUPLE_SEC = 30          # 저하 사건이 끝난 뒤 이 시간 안에 심박 급증이 시작되면 연결된 것으로 본다

ONE_MIN_SPO2 = ["odi_1min", "pct_lt90", "min_spo2", "desat_drop_1min", "desat_min_frac"]
ONE_SEC_SPO2 = ["odi_1s", "pct_lt90_1s", "min_spo2_1s", "desat_drop_1s", "desat_frac_1s"]
ONE_SEC_HR = ["hr_surge_idx_1s", "hr_std5_mean", "desat_hr_coupled_1s"]


def _runs(mask, min_len):
    """True가 min_len 이상 이어지는 구간 [(start, end_exclusive), ...]"""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_len
    return list(zip(starts[keep], ends[keep]))


def spo2_1s_features(x):
    x = np.where((x >= VALID_LO) & (x <= VALID_HI), x, np.nan).astype(float)
    valid = ~np.isnan(x)
    hours = valid.sum() / 3600
    if hours < 1:
        return None, []
    base = pd.Series(x).rolling(SPO2_BASE_SEC, min_periods=SPO2_BASE_SEC // 2).max().shift(1).values
    below = valid & ~np.isnan(base) & (x <= base - DROP_THR)
    ev = _runs(below, SPO2_MIN_SEC)
    depth = [base[a] - np.nanmin(x[a:b]) for a, b in ev]
    return {
        "odi_1s": len(ev) / hours,
        "pct_lt90_1s": 100.0 * float(np.mean(x[valid] < 90)),
        "min_spo2_1s": float(np.percentile(x[valid], 1)),
        "desat_drop_1s": float(np.mean(depth)) if depth else 0.0,
        "desat_frac_1s": 100.0 * sum(b - a for a, b in ev) / valid.sum(),
    }, ev


def hr_1s_features(h, desat_ev):
    h = np.where((h >= HR_LO) & (h <= HR_HI), h, np.nan).astype(float)
    valid = ~np.isnan(h)
    hours = valid.sum() / 3600
    if hours < 1:
        return None
    s = pd.Series(h)
    base = s.rolling(HR_BASE_SEC, min_periods=HR_BASE_SEC // 2).median().shift(1).values
    surge = valid & ~np.isnan(base) & (h >= base + HR_SURGE_BPM)
    ev = _runs(surge, HR_MIN_SEC)
    starts = np.array([a for a, _ in ev])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std5 = s.rolling(300, min_periods=150).std().values[::300]
    if desat_ev:
        hit = [bool(len(starts)) and bool(np.any((starts >= a) & (starts <= b + COUPLE_SEC))) for a, b in desat_ev]
        coupled = 100.0 * float(np.mean(hit))
    else:
        coupled = 0.0
    return {
        "hr_surge_idx_1s": len(ev) / hours,
        "hr_std5_mean": float(np.nanmean(std5)),
        "desat_hr_coupled_1s": coupled,
    }


def subject_features(npz_path):
    d = np.load(npz_path)
    name = os.path.basename(npz_path)[:-4]
    sao2, fs = d["sao2"], float(d["fs"])
    if fs != 1.0:
        return None
    win = shhs_sleep_window(os.path.join(XML_DIR, f"{name}-nsrr.xml"))
    if win:
        sao2 = sao2[int(win[0]): int(win[1])]
    f, ev = spo2_1s_features(sao2)
    if f is None:
        return None
    f["nsrrid"] = int(name.split("-")[1])
    if "hr" in d.files and float(d["hr_fs"]) == 1.0:
        hr = d["hr"][int(win[0]): int(win[1])] if win else d["hr"]
        h = hr_1s_features(hr, ev)
        # 기존 1분 심박 피처 (oximetry_features.hr_features_from_bins)도 같은 구간에서 계산
        sb = bins_from_1hz(sao2, fs)
        hb = bins_from_1hz(hr, 1.0, HR_LO, HR_HI)
        n = min(len(sb[0]), len(hb[0]))
        sb, hb = tuple(x[:n] for x in sb), tuple(x[:n] for x in hb)
        h1 = hr_features_from_bins(*hb, desat_ev=desat_events(sb[0], sb[1], sb[3]))
        if h is not None and h1 is not None:
            f.update(h)
            f.update(h1)
    return f


def build_features():
    if os.path.exists(OUT_CSV):
        return pd.read_csv(OUT_CSV, index_col="nsrrid")
    paths = sorted(glob.glob(os.path.join(SAO2_DIR, "*.npz")))
    rows = []
    for i, p in enumerate(paths, 1):
        f = subject_features(p)
        if f:
            rows.append(f)
        if i % 500 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    df = pd.DataFrame(rows).set_index("nsrrid")
    df.to_csv(OUT_CSV)
    return df


def cv_scores(X, y, repeats=5):
    """(정확도 목록, AUC 목록). 5겹 × repeats회, seed 42부터."""
    accs, aucs = [], []
    for r in range(repeats):
        for tr, te in KFold(5, shuffle=True, random_state=42 + r).split(X):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            accs.append(np.mean((p >= 0.5) == y[te]))
            aucs.append(roc_auc_score(y[te], p))
    return np.array(accs), np.array(aucs)


def report(title, df, y, configs):
    print(f"\n## {title} (n={len(y)}, 양성 {y.mean():.1%})")
    print(f"{'구성':<34}{'정확도':>16}{'AUC':>16}{'Δ정확도':>10}{'ΔAUC':>9}")
    base = None
    for name, cols in configs:
        acc, auc = cv_scores(df[cols].values.astype(float), y)
        if base is None:
            base = (acc, auc)
        print(f"{name:<34}{acc.mean():>9.4f} ± {acc.std():.4f}{auc.mean():>9.4f} ± {auc.std():.4f}"
              f"{(acc - base[0]).mean() * 100:>+9.1f}p{(auc - base[1]).mean():>+9.4f}")


def main():
    cohort = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f"{r}.csv"), index_col="nsrrid") for r in "ABC"], axis=1)
    base16 = [c for c in cohort.columns if c != LABEL]
    feats = build_features()
    df = cohort.join(feats, how="inner")
    y = (df[LABEL].values >= 15).astype(int)
    swapped = [c for c in base16 if c not in ONE_MIN_SPO2] + ONE_SEC_SPO2
    report("SpO2 해상도", df, y, [
        ("16개 (현재, SpO2 1분)", base16),
        ("16개 (SpO2 5개를 1초로 교체)", swapped),
    ])

    hr = df.dropna(subset=ONE_SEC_HR + HR_FEATURES)
    yh = (hr[LABEL].values >= 15).astype(int)
    report("심박 추가 (H.R. 채널 보유자만)", hr, yh, [
        ("16개 (현재)", base16),
        ("16개 + 심박 1분 4개", base16 + HR_FEATURES),
        ("16개 + 심박 1초 3개", base16 + ONE_SEC_HR),
        ("SpO2 1초 16개", swapped),
        ("SpO2 1초 16개 + 심박 1초 3개", swapped + ONE_SEC_HR),
    ])


if __name__ == "__main__":
    main()
