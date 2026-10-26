"""갤럭시 워치(Samsung Health 내보내기) → 밤별 16개 모델 피처 + inference_v5 입력 CSV.

SpO2 1분 피처는 SHHS와 같은 oximetry_features.features_from_bins()로 계산한다.
수면 구간은 SHHS와 같게 첫 수면 단계 시작 ~ 마지막 수면 단계 끝으로 자른다.

워치에 없는 값(성별, 목둘레, ESS)은 인자로 받는다. 키·체중은 weight 기록, 나이는 프로필 생년월일에서 구한다.

사용법:
  python watch_features.py --export 1024/samsunghealth_... --gender 1 --neck 38 --ess 8 \
      [--out-dir inference_v5/watch] [--model-dir inference_v5]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from dataset import MODEL_FEATURES
from oximetry_features import VALID_LO, VALID_HI, HR_LO, HR_HI, features_from_bins

AWAKE = 40001                    # sleep_stage: 40001 깸, 40002 얕은, 40003 깊은, 40004 렘
DEEP = 40003
MIN_SLEEP_MIN = 180              # 낮잠 제외: 총수면 3시간 미만인 세션은 쓰지 않는다
ROLES = {
    "A": ["avgsat", "avg_hr", "odi_1min", "pct_lt90", "min_spo2", "desat_drop_1min", "desat_min_frac"],
    "B": ["slpprdp", "slpeffp", "times34p", "waso"],
    "C": ["age_s1", "gender", "bmi_s1", "neck20", "ess_s1"],
}


def _csv(export, name):
    path = glob.glob(os.path.join(export, f"{name}.*.csv"))[0]
    return pd.read_csv(path, skiprows=1, index_col=False, low_memory=False)


def _ms(series):
    """Samsung Health 시각 문자열(UTC) → epoch ms."""
    return pd.to_datetime(series, utc=True).astype("int64") // 10**6


def _bin_rows(export, kind, table, col_bin, col_start, col_end, t0, t1):
    """[t0, t1)과 겹치는 기록의 1분 binning JSON 행들."""
    rows = []
    hit = table[(table[col_end] > t0) & (table[col_start] < t1) & table[col_bin].notna()]
    for name in hit[col_bin]:
        path = os.path.join(export, "jsons", kind, name[0], name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                rows.extend(r for r in json.load(fh) if t0 <= r["start_time"] < t1)
    return rows


def _minute_grid(rows, t0, n, keys):
    """binning 행 → 수면 구간 시간축의 분 배열들 (빠진 분은 NaN)."""
    out = [np.full(n, np.nan) for _ in keys]
    for r in rows:
        i = int((r["start_time"] - t0) // 60000)
        if 0 <= i < n:
            for arr, k in zip(out, keys):
                arr[i] = r.get(k, np.nan)
    return out


def _profile(export):
    prof = _csv(export, "com.samsung.health.user_profile").set_index("key")
    birth = pd.to_datetime(str(prof.loc["birth_date", "text_value"]), format="%Y%m%d")
    w = _csv(export, "com.samsung.health.weight")
    w["t"] = pd.to_datetime(w["start_time"], utc=True).dt.tz_localize(None)
    height = float(w["height"].dropna().iloc[-1])
    return birth, height, w.dropna(subset=["weight"]).sort_values("t")[["t", "weight"]]


def night_features(export, gender, neck, ess):
    birth, height, weights = _profile(export)
    sleep = _csv(export, "com.samsung.shealth.sleep")
    stages = _csv(export, "com.samsung.health.sleep_stage")
    stages["s"], stages["e"] = _ms(stages["start_time"]), _ms(stages["end_time"])
    spo2 = _csv(export, "com.samsung.shealth.tracker.oxygen_saturation")
    p = "com.samsung.health.oxygen_saturation."
    spo2["s"], spo2["e"] = _ms(spo2[p + "start_time"]), _ms(spo2[p + "end_time"])
    hr = _csv(export, "com.samsung.shealth.tracker.heart_rate")
    q = "com.samsung.health.heart_rate."
    hr["s"], hr["e"] = _ms(hr[q + "start_time"]), _ms(hr[q + "end_time"])

    rows = []
    for _, sess in sleep.iterrows():
        st = stages[stages["sleep_id"] == sess["com.samsung.health.sleep.datauuid"]].sort_values("s")
        asleep = st[st["stage"] != AWAKE]
        if asleep.empty:
            continue
        t0, t1 = int(asleep["s"].min()), int(asleep["e"].max())
        dur = (st["e"].clip(upper=t1) - st["s"].clip(lower=t0)).clip(lower=0) / 60000
        slp = float(dur[st["stage"] != AWAKE].sum())
        if slp < MIN_SLEEP_MIN:
            continue
        n = (t1 - t0) // 60000 + 1

        mean, mn, mx = _minute_grid(_bin_rows(export, "com.samsung.shealth.tracker.oxygen_saturation", spo2,
                                              p + "binning", "s", "e", t0, t1),
                                    t0, n, ("spo2", "spo2_min", "spo2_max"))
        valid = (mean >= VALID_LO) & (mean <= VALID_HI) & (mn >= VALID_LO)
        f = features_from_bins(mean, mn, mx, valid)
        if f is None:
            continue

        # 심박: 깸 단계를 뺀 수면 단계 구간의 측정값 평균. 2025년 이후 밤은 1분 binning이 없고
        # 10분 간격 측정만 있으므로 그 측정값을 쓴다 (최소 1시간 분량 = 6개).
        h = hr[(hr["s"] >= t0) & (hr["s"] < t1)]
        h = h[(h[q + "heart_rate"] >= HR_LO) & (h[q + "heart_rate"] <= HR_HI)]
        in_sleep = np.zeros(len(h), bool)
        for a, b in zip(asleep["s"], asleep["e"]):
            in_sleep |= ((h["s"] >= a) & (h["s"] < b)).values
        if in_sleep.sum() < 6:
            continue

        off = str(sess["com.samsung.health.sleep.time_offset"])  # 예: UTC+0900 → 밤 ID·나이는 현지 시각 기준
        night = pd.Timestamp(t0, unit="ms") + pd.Timedelta(hours=int(off[3:6]), minutes=int(off[3] + off[6:8]))
        wt = weights[weights["t"] <= night]
        weight = float((wt if len(wt) else weights).iloc[-1 if len(wt) else 0]["weight"])
        rows.append({
            "nsrrid": night.strftime("%Y%m%d%H%M"),
            "avgsat": f["avg_spo2"], "avg_hr": float(h[q + "heart_rate"][in_sleep].mean()),
            "odi_1min": f["odi_1min"], "pct_lt90": f["pct_lt90"], "min_spo2": f["min_spo2"],
            "desat_drop_1min": f["desat_drop_1min"], "desat_min_frac": f["desat_min_frac"],
            "slpprdp": slp, "slpeffp": float(sess["efficiency"]),
            "times34p": 100.0 * float(dur[st["stage"] == DEEP].sum()) / slp,
            "waso": float(dur[st["stage"] == AWAKE].sum()),
            "age_s1": (night - birth).days // 365.25, "gender": gender,
            "bmi_s1": weight / (height / 100) ** 2, "neck20": neck, "ess_s1": ess,
            "valid_hours": f["valid_hours"],
        })
    return pd.DataFrame(rows).set_index("nsrrid").sort_index()


def _tokens(path):
    with open(path, encoding="utf-8") as fh:
        return {l.split()[0]: l.split()[1:] for l in fh.read().splitlines()[1:] if l.strip()}


def predict(df, model_dir):
    """inference_v5 모델 파일로 평문 추론 (CKKS 결과와 오차 1e-6 수준)."""
    c = _tokens(os.path.join(model_dir, "client_model.txt"))
    names = c["FEATURE_NAMES"]
    mu, sc = np.array(c["SCALER_MEAN"], float), np.array(c["SCALER_SCALE"], float)
    w = np.full(len(names), np.nan)
    for h in range(3):
        m = _tokens(os.path.join(model_dir, f"hospital{h}_model.txt"))
        w[list(map(int, m["INDICES"]))] = list(map(float, m["WEIGHTS"]))
    zx = (df[names].values - mu) / sc
    logit = zx @ w + float(c["BIAS"][0])
    return pd.DataFrame(zx * w, index=df.index, columns=names), logit, 1 / (1 + np.exp(-logit))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Samsung Health 내보내기 폴더")
    ap.add_argument("--gender", type=int, required=True, choices=[1, 2])
    ap.add_argument("--neck", type=float, required=True)
    ap.add_argument("--ess", type=float, required=True)
    ap.add_argument("--out-dir", default="inference_v5/watch")
    ap.add_argument("--model-dir", default="inference_v5")
    a = ap.parse_args()

    df = night_features(a.export, a.gender, a.neck, a.ess)
    assert sorted(sum(ROLES.values(), [])) == sorted(MODEL_FEATURES)
    os.makedirs(a.out_dir, exist_ok=True)
    for role, cols in ROLES.items():
        out = df[cols].copy()
        if role == "B":
            out["ahi_a0h3a"] = -1  # 워치에는 라벨이 없다 (DistributedInferenceV5.cpp가 열을 요구함)
        out.to_csv(os.path.join(a.out_dir, f"{role}.csv"))
    print(f"[watch] {len(df)}밤 → {a.out_dir}/A.csv, B.csv, C.csv")

    contrib, logit, prob = predict(df, a.model_dir)
    df["logit"], df["prob"] = logit, prob
    df.to_csv(os.path.join(a.out_dir, "watch_predictions.csv"))
    print(df[MODEL_FEATURES].median().round(2).to_string())
    print(f"\nP(AHI>=15): median={np.median(prob):.3f}  min={prob.min():.3f}  max={prob.max():.3f}  "
          f">=0.5: {(prob >= 0.5).sum()}/{len(prob)}밤")
    print("\nlogit 기여도 중앙값 (w * 표준화 값):")
    print(contrib.median().sort_values().round(3).to_string())
