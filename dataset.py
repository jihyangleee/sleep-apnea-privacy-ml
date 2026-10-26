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
    # feature 확장 실험 (워치/앱으로 수집 가능한 SHHS 변수만 선정)
    "waso",           # wake after sleep onset (분) — actigraphy로 워치가 추정 가능
    "neck20",         # 목둘레 — 앱 1회 수동 입력
    "ess_s1",         # Epworth 졸림 점수 — 앱 온보딩 설문(자기보고)
]
# CSV에서 avg_hr을 계산할 때 사용하는 4개 컬럼 (NREM/REM × 앙와위/비앙와위)
_SHHS_HR_COLS = ["savbnbh", "savbnoh", "savbrbh", "savbroh"]
SHHS_LABEL    = "ahi_a0h3a"
AHI_THRESHOLD = 15  # 중등도 이상 수면무호흡

# SpO2 1분 피처 — EDF SaO2(1Hz)를 워치와 같은 1분 (평균, 최소, 최대)로 묶어 계산한다
# (oximetry_features.py, 워치 데이터에도 같은 함수를 쓴다).
# 예전 XML 저하 피처(odi, mean_desat_duration, mean_desat_drop)는 무호흡·저호흡에 딸린 저하만 담고
# 있어 라벨(AHI) 누출이 생기므로 쓰지 않는다 (extract_desat_features.py는 기록용으로만 남김).
SHHS_OXI_FEATURES = ["odi_1min", "pct_lt90", "min_spo2", "desat_drop_1min", "desat_min_frac"]
# 학습·추론에 실제로 쓰는 전체 피처 순서 (모델 입력 = 이 순서)
MODEL_FEATURES = SHHS_FEATURES + SHHS_OXI_FEATURES   # 11 + 5 = 16
DEFAULT_XML_DIR    = "shhs/polysomnography/annotations-events-nsrr/shhs1"
OXI_FEATURES_CSV   = "shhs/datasets/oximetry_1min_features.csv"  # python oximetry_features.py 결과


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


def _load_oximetry_features(csv_path: str = OXI_FEATURES_CSV) -> pd.DataFrame:
    """nsrrid로 색인된 SpO2 1분 피처 표. 없으면 `python oximetry_features.py`로 먼저 만든다
    (EDF SaO2는 download_shhs_sao2.py가 shhs/sao2/에 받아 둔다)."""
    import os
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} 없음 — 먼저 `python oximetry_features.py` 실행")
    return pd.read_csv(csv_path, index_col="nsrrid")[SHHS_OXI_FEATURES]


def load_shhs_data(csv_path: str, return_scaler: bool = False, oxi_csv: str = OXI_FEATURES_CSV):
    """NSRR SHHS-1 CSV + SpO2 1분 피처 로드 → (X, y) 반환.

    X — (n_samples, 16) float32, StandardScaler 정규화. 열 순서는 MODEL_FEATURES
        (CSV 11개 + SpO2 1분 5개). EDF SaO2가 있는 참가자만 남는다(inner join).
    y — (n_samples,)   float32, 0=정상 / 1=수면무호흡(AHI≥15)
    return_scaler=True 시 (X, y, scaler) 반환
    """
    csv_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL, "nsrrid"]
    df = pd.read_csv(csv_path, usecols=csv_cols)
    df["avg_hr"] = df[_SHHS_HR_COLS].mean(axis=1)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    # avg_hr은 4개 HR 열의 (결측 무시) 평균이므로 원시 HR 열의 결측은 버릴 이유가 없다.
    # df.dropna() 전체를 쓰면 savbrbh 등이 비어 있는 참가자를 불필요하게 잃는다(5.8천 → 2.5천명).
    df = df.dropna(subset=SHHS_FEATURES + [SHHS_LABEL])
    df = df.set_index("nsrrid").join(_load_oximetry_features(oxi_csv), how="inner")

    X = df[MODEL_FEATURES].values.astype(np.float32)
    y = (df[SHHS_LABEL].values >= AHI_THRESHOLD).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[SHHS] {len(X)}명 로드 (피처 {X.shape[1]}개) | 수면무호흡(AHI≥{AHI_THRESHOLD}): {int(y.sum())}명 ({y.mean():.1%})")
    if return_scaler:
        return X, y, scaler
    return X, y


# 갤럭시 워치 사용자 합성 프로파일 (원시 단위값, 정규화 전) <-- 더미 값
# feature 순서 = MODEL_FEATURES:
#   avgsat, avg_hr, slpprdp, slpeffp, times34p, age_s1, gender, bmi_s1,
#   waso, neck20, ess_s1, odi, mean_desat_duration, mean_desat_drop
_GALAXY_WATCH_PROFILES = [
    {"name": "고위험 | 54세 남성 | 비만·저산소",        "raw": [91.0, 74.0, 310.0, 73.0,  9.0, 54.0, 1.0, 33.5, 75.0, 43.0, 13.0, 38.0, 26.0, 5.5]},
    {"name": "중위험 | 44세 남성 | 과체중",              "raw": [94.0, 68.0, 355.0, 81.0, 13.0, 44.0, 1.0, 27.5, 45.0, 39.0,  9.0, 16.0, 22.0, 3.0]},
    {"name": "저위험 | 32세 여성 | 정상",                "raw": [97.0, 61.0, 425.0, 91.0, 21.0, 32.0, 2.0, 21.5, 20.0, 32.0,  5.0,  3.0, 18.0, 1.8]},
    {"name": "고위험 | 61세 남성 | 고도비만·심한 저산소", "raw": [88.0, 79.0, 275.0, 69.0,  7.0, 61.0, 1.0, 36.0, 90.0, 45.0, 16.0, 52.0, 30.0, 7.5]},
    {"name": "저위험 | 28세 여성 | 활동적",              "raw": [96.5, 57.0, 450.0, 93.0, 23.0, 28.0, 2.0, 20.5, 15.0, 31.0,  4.0,  2.0, 17.0, 1.5]},
]


def generate_galaxy_watch_users():
    """갤럭시 워치 사용자 합성 프로파일 5명 반환 (정규화 전 원시값).

    반환:
        raw_X  — (5, 14) float32, 정규화 전 원시 단위값
        names  — list[str], 프로파일 설명
    """
    raw_X = np.array([p["raw"] for p in _GALAXY_WATCH_PROFILES], dtype=np.float32)
    names = [p["name"] for p in _GALAXY_WATCH_PROFILES]
    return raw_X, names


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
            "feature_indices": fg,
        })
    return partitions, X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test


