"""SHHS 통합 CSV+XML을 병원별 로컬 파일 3개로 쪼갠다 (일회성 준비 단계).

지금까지는 한 프로세스가 dataset.load_shhs_data()로 14개 피처 전부를 읽은 뒤
메모리 안에서 열(column) 기준으로 나눴다. 이건 시뮬레이션이라 괜찮았지만,
실제로는 병원 A가 병원 B·C의 원본 컬럼을 단 한 번도 봐서는 안 된다.

이 스크립트는 "최초 1회, 각 병원이 자기 쪽 원천 시스템에서 직접 내보내는 것"과
동일한 효과를 내도록, 통합 CSV를 미리 3개 파일로 쪼개 둔다. 쪼갠 뒤에는
hospital_a.py / hospital_b.py / hospital_c.py 가 각자 자기 파일만 읽는다 --
다른 병원 파일은 아예 경로조차 모른다.

라벨(y)은 PSG를 직접 수행해 AHI를 산출하는 기관이 원본을 쥐고 있다는 논문
6.4절의 설명에 맞춰, 병원 B(Sleep lab, PSG)의 파일에 넣는다. (기존
SLEEP_CLIENT_LABELS 텍스트는 병원 C라고 적어 뒀었는데, "PSG 수행 기관이 라벨을
안다"는 논문 본문 설명과는 병원 B가 더 맞아서 여기서 B로 통일한다 -- 텍스트
라벨 쪽을 고쳐야 한다는 뜻이기도 하다.)
학습 프로토콜 자체는 y를 additive_split으로 다시 비밀분산하므로, "원본을 아는 것"과
"학습 중 평문으로 보유하는 것"은 다른 문제다 (6.4절의 라벨 입력 가정과 동일).

출력: mpc/hospital_data/{A,B,C}.csv  (각자 nsrrid 기준으로 정렬됨)

사용법: PYTHONPATH=<프로젝트 루트> python mpc/prepare_hospital_data.py [csv]
"""

import os
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import (
    SHHS_FEATURES, SHHS_LABEL, _SHHS_HR_COLS, _load_oximetry_features,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hospital_data")

ROLE_FEATURES = {
    "A": ["avgsat", "avg_hr", "odi_1min", "pct_lt90", "min_spo2", "desat_drop_1min", "desat_min_frac"],
    "B": ["slpprdp", "slpeffp", "times34p", "waso"],
    "C": ["age_s1", "gender", "bmi_s1", "neck20", "ess_s1"],
}
LABEL_HOLDER = "B"  # PSG를 수행해 AHI를 직접 산출하는 기관


def main(csv_path):
    csv_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL, "nsrrid"]
    df = pd.read_csv(csv_path, usecols=csv_cols)
    df["avg_hr"] = df[_SHHS_HR_COLS].mean(axis=1)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna(subset=SHHS_FEATURES + [SHHS_LABEL])
    df = df.set_index("nsrrid").join(_load_oximetry_features(), how="inner")
    df = df.sort_index()

    os.makedirs(OUT_DIR, exist_ok=True)
    for role, feats in ROLE_FEATURES.items():
        cols = feats + ([SHHS_LABEL] if role == LABEL_HOLDER else [])
        out_path = os.path.join(OUT_DIR, f"{role}.csv")
        df[cols].to_csv(out_path)
        print(f"[{role}] {out_path}  (피처 {len(feats)}개"
              + (" + label" if role == LABEL_HOLDER else "") + f", n={len(df)})")

    print("\n주의: 이 스크립트는 '최초 준비 단계'를 시뮬레이션한 것이다. 실제 배포에서는 이 통합 ")
    print("단계 자체가 없어야 하고, 각 병원이 직접 자기 원천 DB에서 자기 컬럼만 내보내야 한다.")
    print("지금은 세 파일이 전부 nsrrid로 미리 정렬돼 있다고 가정했는데, 실제로는 병원마다 환자")
    print("ID 체계가 다르므로 PSI(Private Set Intersection)로 정렬을 맞추는 단계가 먼저 필요하다")
    print("(다음 작업 항목).")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    main(csv_path)
