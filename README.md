# Sleep Apnea Federated Learning with Homomorphic Encryption

수면 무호흡증 예측을 위한 **Vertical Federated Learning** + **동형암호화(HE) 추론** 시스템.

여러 의료 기관이 원시 데이터를 공유하지 않고 협력하여 모델을 학습하고,
학습에 참여하지 않은 신규 개인도 자신의 데이터를 암호화한 채로 예측 결과를 받을 수 있다.

---

## 전체 구조

```
[학습 단계 — Vertical Federated Learning]

  client A (심박·산소 모니터링)   client B (수면검사실 PSG)   client C (병원/클리닉)
  SpO2, resting HR, avg HR       total sleep, efficiency,    age, sex, BMI
                                  deep sleep ratio            + 레이블(수면무호흡 여부)
         ↓ sub_model_A                  ↓ sub_model_B               ↓ sub_model_C
      embedding_A (16d)            embedding_B (16d)           embedding_C (16d)
                    └──────────────────┴────────────────────────────┘
                                       ↓ concat (48d)
                                  [서버 top_model]
                              Linear(48→32) → Square → Linear(32→1)
                                       ↓ BCEWithLogitsLoss → 역전파


[추론 단계 — HE Inference]

  신규 개인(환자)                              서버
  feature 9개 전체                             평문 가중치 보유
       ↓ CKKS 암호화                           공개키만 사용
  암호화된 벡터 ──────────────────────────→  HE 연산 (ciphertext × plaintext)
                                               ↓ 암호화된 로짓
  예측 확률 ←─────── 복호화 (비밀키) ────────  암호화된 결과 반환
```

---

## 데이터셋

**Sleep Heart Health Study (SHHS-1)** — NSRR (National Sleep Research Resource)

- 출처: [sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs)
- 접근: 연구자 승인 신청 후 다운로드 (`shhs1-dataset-*.csv`)
- 레이블: `ahi_a0h3a ≥ 15` → 중등도 이상 수면무호흡 (1), 정상 (0)

| 피처 | SHHS 변수명 | 담당 기관 |
|---|---|---|
| SpO2 (평균 산소포화도) | `avgsat` | client A |
| Resting heart rate | `avhr` | client A |
| Avg heart rate | `avhrbk` | client A |
| Total sleep time (분) | `slpprdp` | client B |
| Sleep efficiency (%) | `slpeffic` | client B |
| Deep sleep ratio (%) | `pctsa34p` | client B |
| Age | `age_s1` | client C |
| Sex | `gender` | client C |
| BMI | `bmi_s1` | client C |

더 많은 Feature가 존재하지만 Galaxy Watch와 연동하여 수면 무호흡증을 예측하기 위해 위의 Feature만 사용함

---

## 모델 아키텍처

### ClientSubModel (`model.py`)
각 클라이언트가 담당 feature를 임베딩으로 변환하는 서브모델.

```
Linear(input_dim → 16) → Square(x²)
```

- 활성화 함수로 **ReLU/Sigmoid 대신 x²(Square)** 사용
- 이유: CKKS 동형암호는 다항식 연산만 지원하므로, HE 추론과 호환되는 활성화 함수 필요

### ServerTopModel (`model.py`)
서버가 보유하는 탑모델. 연결된 임베딩으로 최종 예측.

```
Linear(48 → 32) → Square(x²) → Linear(32 → 1)
```

- Sigmoid 없음 (HE 호환, 학습 시 BCEWithLogitsLoss 사용)

### VerticalHeartNet (`model.py`)
서브모델 3개 + 탑모델을 통합한 전체 모델. 저장/로드 및 HE 추론에 사용.

---

## HE 추론 설계 (`he_client.py`)

### 암호화 스킴: CKKS (Cheon-Kim-Kim-Song)
근사 실수 연산을 지원하는 동형암호 스킴. 신경망 추론에 적합.

### CKKS 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `poly_modulus_degree` | 8192 | 보안 수준 및 슬롯 수 결정 |
| `coeff_mod_bit_sizes` | [60, 40, 40, 60] | 3레벨(곱셈 횟수) 제공 |
| `global_scale` | 2⁴⁰ | 실수 정밀도 |

### 곱셈 깊이 분석
HE는 암호문끼리의 곱셈(Square 활성화)마다 레벨을 1씩 소비한다.

```
서브모델 Square: 1레벨 소비
탑모델  Square: 1레벨 소비
─────────────────────────
합계           : 2레벨 소비  ←  제공 3레벨로 충분
```

### 가중치 확장(zero-padding) 전략
단일 암호문 벡터(9차원) 하나로 모든 서브모델의 선형 변환을 처리하기 위해,
각 서브모델의 가중치를 전체 feature 크기(9 × 48)로 확장하고 담당하지 않는 위치는 0으로 채운다.

```
W_ext_A: (9, 48), 열 [0:16]만 유효
W_ext_B: (9, 48), 열 [16:32]만 유효
W_ext_C: (9, 48), 열 [32:48]만 유효

enc_emb_A = (enc_x @ W_ext_A + b_ext_A)²   → 위치 [0:16]만 non-zero
enc_emb_B = (enc_x @ W_ext_B + b_ext_B)²   → 위치 [16:32]만 non-zero
enc_emb_C = (enc_x @ W_ext_C + b_ext_C)²   → 위치 [32:48]만 non-zero

enc_concat = enc_emb_A + enc_emb_B + enc_emb_C  ← 구간 비중복이므로 합 = 연결
```

---

## 사용 라이브러리

| 라이브러리 | 버전 권장 | 용도 |
|---|---|---|
| `torch` (PyTorch) | ≥ 2.0 | 모델 정의, 학습, 역전파 |
| `tenseal` | ≥ 0.3 | CKKS 동형암호 (Microsoft SEAL 래퍼) |
| `numpy` | ≥ 1.24 | 행렬 연산, HE 가중치 확장 |
| `pandas` | ≥ 2.0 | SHHS CSV 로드 |
| `scikit-learn` | ≥ 1.3 | StandardScaler, train_test_split |

---

## 파일 구조

```
.
├── dataset.py      데이터 로드(SHHS) 및 Vertical FL용 column 분할
├── model.py        ClientSubModel, ServerTopModel, VerticalHeartNet
├── client.py       VerticalClient — 서브모델 학습 및 임베딩 계산
├── server.py       VerticalFLServer — 탑모델 학습 및 정확도 평가
├── simulate.py     Vertical FL 시뮬레이션 루프 (feature 그룹 정의 포함)
├── he_client.py    HEInference — CKKS 암호화 입력으로 프라이빗 추론
└── main.py         진입점 (--mode vertical / he-infer, --csv 경로)
```

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install torch tenseal numpy pandas scikit-learn
```

### 2. SHHS 데이터 준비

[sleepdata.org](https://sleepdata.org/datasets/shhs) 에서 연구자 승인 후 `shhs1-dataset-*.csv` 다운로드.


### 3. Vertical FL 학습

```bash
python main.py --mode vertical --csv shhs1-dataset-0.21.0.csv
```

학습 완료 후 `vertical_model.pt` 저장됨.

### 4. HE 추론 데모

```bash
python main.py --mode he-infer --csv shhs1-dataset-0.21.0.csv
```

평문 추론 결과와 HE 추론 결과를 비교 출력. 오차 < 0.01이면 ✓.

---

## 프라이버시 보장 범위

| | 보호 여부 |
|---|---|
| 각 기관의 원시 feature (학습 중) | △ 임베딩만 서버에 전달 (Split Learning 수준) |
| 신규 개인의 입력 데이터 (추론 중) | ✅ 서버가 평문을 볼 수 없음 (CKKS HE) |
| 추론 결과 | ✅ 암호화된 상태로 반환, 개인만 복호화 가능 |
| 모델 가중치 | ❌ 서버가 평문으로 보유 |
