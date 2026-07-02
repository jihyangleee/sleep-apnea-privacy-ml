from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# NSRR SHHS-1 변수명 — 실제 CSV 컬럼명과 다를 경우 여기서만 수정
SHHS_FEATURES = [
    "avgsat",     # SpO2: 평균 산소포화도 (%) — CSV: avgsat
    "avg_hr",     # avg heart rate: 평균 심박수 — CSV에 없음, load 시 4개 컬럼 평균으로 계산
    "slpprdp",    # total sleep time: 총 수면 시간 (분) — CSV: slpprdp
    "slpeffp",    # sleep efficiency: 수면 효율 (%) — CSV: slpeffp
    "times34p",   # deep sleep ratio: 3-4단계 수면 비율 (%) — CSV: times34p
    "age_s1",     # age
    "gender",     # sex: 1=남, 2=여
    "bmi_s1",     # BMI
]
# CSV에서 avg_hr을 계산할 때 사용하는 4개 컬럼 (NREM/REM × 앙와위/비앙와위)
_SHHS_HR_COLS = ["savbnbh", "savbnoh", "savbrbh", "savbroh"]
SHHS_LABEL    = "ahi_a0h3a"
AHI_THRESHOLD = 15  # 중등도 이상 수면무호흡


def explore_shhs_csv(csv_path: str):
    """CSV 받은 직후 실행 — 실제 컬럼명 확인 및 SHHS_FEATURES 매핑 검증."""
    df = pd.read_csv(csv_path, nrows=5)
    print("=== 전체 컬럼 목록 ===")
    for i, col in enumerate(df.columns):
        print(f"  {i:3d}  {col}")
    print("\n=== SHHS_FEATURES 매핑 검증 ===")
    check_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL]
    for name in check_cols:
        status = "✅" if name in df.columns else "❌ 없음 — 컬럼명 확인 필요"
        print(f"  {name:<15} {status}")


def load_shhs_data(csv_path: str, return_scaler: bool = False):
    """NSRR SHHS-1 CSV 로드 → (X, y) 반환.

    X — (n_samples, 8) float32, StandardScaler 정규화
    y — (n_samples,)   float32, 0=정상 / 1=수면무호흡(AHI≥15)
    return_scaler=True 시 (X, y, scaler) 반환
    """
    csv_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL]
    df = pd.read_csv(csv_path, usecols=csv_cols)
    df["avg_hr"] = df[_SHHS_HR_COLS].mean(axis=1)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna()

    X = df[SHHS_FEATURES].values.astype(np.float32)
    y = (df[SHHS_LABEL].values >= AHI_THRESHOLD).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[SHHS] {len(X)}명 로드 | 수면무호흡(AHI≥{AHI_THRESHOLD}): {int(y.sum())}명 ({y.mean():.1%})")
    if return_scaler:
        return X, y, scaler
    return X, y


def generate_dummy_data(n_samples: int = 1000, random_seed: int = 42, return_scaler: bool = False):
    """실제 CSV 없이 테스트용 합성 데이터 생성.

    SHHS 피처 분포를 근사한 임의 데이터.
    X — (n_samples, 8) float32, StandardScaler 정규화
    y — (n_samples,)   float32, 0/1 레이블 (BMI·나이·SpO2 기반 확률 할당)
    return_scaler=True 시 (X, y, scaler) 반환
    """
    rng = np.random.default_rng(random_seed)
    avgsao2   = rng.normal(95.0, 2.5,  n_samples).clip(70, 100)   # SpO2 (%)
    avg_hr    = rng.normal(65.0, 10.0, n_samples).clip(40, 120)   # avg HR (bpm)
    slptime   = rng.normal(380.0, 70.0, n_samples).clip(60, 600)  # total sleep (min)
    slp_eff   = rng.normal(85.0, 10.0, n_samples).clip(20, 100)   # sleep eff (%)
    timest34p = rng.normal(15.0, 8.0,  n_samples).clip(0,  50)    # deep sleep (%)
    age       = rng.uniform(30, 75,    n_samples)                  # age
    gender    = rng.choice([1, 2],     n_samples).astype(float)   # 1=male 2=female
    bmi       = rng.normal(28.0, 6.0,  n_samples).clip(15, 55)    # BMI

    X = np.column_stack([avgsao2, avg_hr, slptime, slp_eff, timest34p, age, gender, bmi]).astype(np.float32)

    # 수면무호흡 확률: BMI↑ + 나이↑ + SpO2↓ + 남성 에 비례
    score = (bmi - 28) * 0.05 + (age - 50) * 0.02 + (95 - avgsao2) * 0.08 + (gender == 1) * 0.3
    prob  = 1 / (1 + np.exp(-score))
    y = (rng.random(n_samples) < prob).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print(f"[Dummy] {n_samples}명 생성 | 수면무호흡: {int(y.sum())}명 ({y.mean():.1%})")
    if return_scaler:
        return X, y, scaler
    return X, y


# 갤럭시 워치 사용자 합성 프로파일 (원시 단위값, 정규화 전)
# feature 순서: avgsao2, avg_hr, slptime, slp_eff, timest34p, age, gender, bmi
_GALAXY_WATCH_PROFILES = [
    {"name": "고위험 | 54세 남성 | 비만·저산소",        "raw": [91.0, 74.0, 310.0, 73.0,  9.0, 54.0, 1.0, 33.5]},
    {"name": "중위험 | 44세 남성 | 과체중",              "raw": [94.0, 68.0, 355.0, 81.0, 13.0, 44.0, 1.0, 27.5]},
    {"name": "저위험 | 32세 여성 | 정상",                "raw": [97.0, 61.0, 425.0, 91.0, 21.0, 32.0, 2.0, 21.5]},
    {"name": "고위험 | 61세 남성 | 고도비만·심한 저산소", "raw": [88.0, 79.0, 275.0, 69.0,  7.0, 61.0, 1.0, 36.0]},
    {"name": "저위험 | 28세 여성 | 활동적",              "raw": [96.5, 57.0, 450.0, 93.0, 23.0, 28.0, 2.0, 20.5]},
]


def generate_galaxy_watch_users():
    """갤럭시 워치 사용자 합성 프로파일 5명 반환 (정규화 전 원시값).

    반환:
        raw_X  — (5, 8) float32, 정규화 전 원시 단위값
        names  — list[str], 프로파일 설명
    """
    raw_X = np.array([p["raw"] for p in _GALAXY_WATCH_PROFILES], dtype=np.float32)
    names = [p["name"] for p in _GALAXY_WATCH_PROFILES]
    return raw_X, names


def load_dreamt_data(dreamt_dir: str, return_scaler: bool = False):
    """DREAMT v2.1.0 로드 → (X, y) 반환.

    dreamt_dir: physionet.org/files/dreamt/2.1.0/ 경로
    X — (n_samples, 8) float32, StandardScaler 정규화
    y — (n_samples,)   float32, 0=정상 / 1=수면무호흡(AHI≥15)

    feature 순서: avgsao2, avg_hr, slptime, slp_eff, timest34p, age, gender, bmi
    - avgsao2, age, gender, bmi → participant_info.csv에서 직접 읽음
    - avg_hr, slptime, slp_eff, timest34p → data_64Hz/SID_whole_df.csv 신호 집계
    """
    dreamt_dir = Path(dreamt_dir)
    signal_dir = dreamt_dir / "data_64Hz"

    info = pd.read_csv(dreamt_dir / "participant_info.csv")
    info["Mean_SaO2"] = info["Mean_SaO2"].str.rstrip("%").astype(float)
    info["GENDER_NUM"] = info["GENDER"].map({"M": 1.0, "F": 2.0})

    SLEEP_STAGES = {"N1", "N2", "N3", "R"}

    rows = []
    missing_files = []
    for _, row in info.iterrows():
        sid = row["SID"]
        sig_path = signal_dir / f"{sid}_whole_df.csv"
        if not sig_path.exists():
            missing_files.append(sid)
            continue

        sig = pd.read_csv(sig_path, usecols=["HR", "Sleep_Stage"])
        sleep_mask = sig["Sleep_Stage"].isin(SLEEP_STAGES)
        bed_mask   = ~sig["Sleep_Stage"].isin(["P", "Missing"])
        n3_mask    = sig["Sleep_Stage"] == "N3"

        sleep_n = sleep_mask.sum()
        bed_n   = bed_mask.sum()
        if sleep_n == 0 or bed_n == 0:
            continue

        # HR은 1Hz 샘플링 → NaN 행 무시하고 수면 구간 평균
        avg_hr    = sig.loc[sleep_mask, "HR"].dropna().mean()
        slptime   = sleep_n / 64 / 60                       # 분
        slp_eff   = slptime / (bed_n / 64 / 60) * 100      # %
        timest34p = n3_mask.sum() / sleep_n * 100           # %

        rows.append({
            "avgsao2":   row["Mean_SaO2"],
            "avg_hr":    avg_hr,
            "slptime":   slptime,
            "slp_eff":   slp_eff,
            "timest34p": timest34p,
            "age_s1":    row["AGE"],
            "gender":    row["GENDER_NUM"],
            "bmi_s1":    row["BMI"],
            "ahi":       row["AHI"],
        })

    if missing_files:
        print(f"[DREAMT] 신호 파일 없음 ({len(missing_files)}명 제외): {missing_files[:5]}{'...' if len(missing_files)>5 else ''}")
    if not rows:
        raise FileNotFoundError(
            f"[DREAMT] data_64Hz/ 신호 파일을 찾을 수 없습니다. (dreamt_dir={dreamt_dir})\n"
            "wget 다운로드가 완료됐는지 확인하세요."
        )

    df = pd.DataFrame(rows).dropna()
    X  = df[["avgsao2", "avg_hr", "slptime", "slp_eff", "timest34p", "age_s1", "gender", "bmi_s1"]].values.astype(np.float32)
    y  = (df["ahi"].values >= AHI_THRESHOLD).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print(f"[DREAMT] {len(X)}명 로드 | 수면무호흡(AHI≥{AHI_THRESHOLD}): {int(y.sum())}명 ({y.mean():.1%})")
    if return_scaler:
        return X, y, scaler
    return X, y


def partition_data_vertical(X, y, num_clients: int, test_size: float = 0.2, feature_groups: list = None):
    """데이터를 열(feature) 기준으로 분할 — Vertical FL용."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    if feature_groups is None:
        feature_groups = [arr.tolist() for arr in np.array_split(np.arange(X.shape[1]), num_clients)]

    partitions = []
    for fg in feature_groups:
        partitions.append({
            "X_train": X_train[:, fg].astype(np.float32),
            "X_test":  X_test[:, fg].astype(np.float32),
            "y_train": y_train,
            "y_test":  y_test,
            "feature_indices": fg,
        })
    return partitions, X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test


