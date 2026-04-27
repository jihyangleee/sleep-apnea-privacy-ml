import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# NSRR SHHS-1 데이터셋 변수명
# 실제 다운로드한 CSV 파일의 컬럼명과 다를 경우 여기서만 수정
SHHS_FEATURES = [
    "avgsat",    # SpO2: 평균 산소포화도 (%)
    "avhr",      # resting heart rate: 안정 시 심박수
    "avhrbk",    # avg heart rate: 평균 심박수
    "slpprdp",   # total sleep time: 총 수면 시간 (분)
    "slpeffic",  # sleep efficiency: 수면 효율 (%)
    "pctsa34p",  # deep sleep ratio: 3-4단계 수면 비율 (%)
    "age_s1",    # age
    "gender",    # sex: 1=남, 2=여
    "bmi_s1",    # BMI
]
SHHS_LABEL    = "ahi_a0h3a"   # AHI (3% 산소포화도 기준)
AHI_THRESHOLD = 15            # 15 이상 = 중등도 이상 수면무호흡


def load_shhs_data(csv_path: str):
    """NSRR SHHS-1 CSV 파일을 로드하여 (X, y)를 반환.

    csv_path: 다운로드한 shhs1-dataset-*.csv 경로
    반환:
      X — (n_samples, 9) float32, StandardScaler 정규화 완료
      y — (n_samples,)   float32, 0=정상 / 1=중등도 이상 수면무호흡(AHI≥15)
    """
    df = pd.read_csv(csv_path, usecols=SHHS_FEATURES + [SHHS_LABEL])

    # KNOWNISSUES: 불가능한 값 제거 (음수 AHI, 0% 수면효율 등)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffic"] > 0]
    df = df.dropna()

    X = df[SHHS_FEATURES].values.astype(np.float32)
    y = (df[SHHS_LABEL].values >= AHI_THRESHOLD).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[SHHS] 로드 완료 — 총 {len(X)}명 | 수면무호흡(AHI≥{AHI_THRESHOLD}): {int(y.sum())}명 ({y.mean():.1%})")
    return X, y


def partition_data_vertical(X, y, num_clients: int, test_size: float = 0.2, feature_groups: list = None):
    """데이터를 열(feature) 기준으로 분할 — Vertical FL용.
    모든 클라이언트가 동일한 샘플(row)을 보유하고 서로 다른 feature(column)를 담당한다.
    feature_groups를 지정하지 않으면 균등 분할한다.
    """
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
