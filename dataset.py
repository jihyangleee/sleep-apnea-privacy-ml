"""SHHS 피처·라벨 정의. 학습(mpc/prepare_hospital_data.py)과 추론 모델 내보내기(mpc/export_heaan2.py),
워치 피처 계산(watch_features.py)이 같은 피처 이름과 순서를 쓰도록 여기 한 곳에 둔다."""

import os

import pandas as pd

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
# 있어 라벨(AHI) 누출이 생기므로 쓰지 않는다.
SHHS_OXI_FEATURES = ["odi_1min", "pct_lt90", "min_spo2", "desat_drop_1min", "desat_min_frac"]
# 학습·추론에 실제로 쓰는 전체 피처 순서 (모델 입력 = 이 순서)
MODEL_FEATURES = SHHS_FEATURES + SHHS_OXI_FEATURES   # 11 + 5 = 16
OXI_FEATURES_CSV   = "shhs/datasets/oximetry_1min_features.csv"  # python oximetry_features.py 결과


def _load_oximetry_features(csv_path: str = OXI_FEATURES_CSV) -> pd.DataFrame:
    """nsrrid로 색인된 SpO2 1분 피처 표. 없으면 `python oximetry_features.py`로 먼저 만든다
    (EDF SaO2는 download_shhs_sao2.py가 shhs/sao2/에 받아 둔다)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} 없음 — 먼저 `python oximetry_features.py` 실행")
    return pd.read_csv(csv_path, index_col="nsrrid")[SHHS_OXI_FEATURES]

