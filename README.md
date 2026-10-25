# Sleep Apnea VFL + MPC + HE

갤럭시 워치로 수집할 수 있는 데이터로 **중등도 이상 수면무호흡(AHI ≥ 15)** 위험을 예측하는 시스템이다.

- **학습**: 세 병원이 각자 가진 피처를 서로 공개하지 않고 함께 학습한다. 이를 위해 수직 연합학습(Vertical FL)과 SecretFlow SPU 기반 MPC, DP 노이즈를 쓴다.
- **추론**: 환자 기기가 피처를 CKKS로 암호화해 병원에 보낸다. 병원은 암호문 상태로 계산하고, 결과는 비밀키를 가진 환자만 복호화한다. 구현은 HEaaN2(C++)를 쓴다.

---

## 1. 데이터 변수

### 학습 데이터

**Sleep Heart Health Study 1차 방문(SHHS1)**을 NSRR에서 받아 쓴다.

| 출처 | 경로 | 용도 |
|---|---|---|
| SHHS CSV | `shhs/datasets/shhs1-dataset-0.21.0.csv` | 인구통계, 수면 지표, 라벨 |
| PSG 원신호 (EDF) | `shhs/sao2/*.npz` | SaO₂ 채널(1Hz)만 추출해 저장 (`download_shhs_sao2.py`) |
| 사건 주석 (XML) | `shhs/polysomnography/annotations-events-nsrr/shhs1/` | 수면 단계만 사용 (수면 구간 자르기) |

- **라벨**: `ahi_a0h3a ≥ 15` → 1 (중등도 이상 수면무호흡), 미만 → 0
- 변수 정의: [NSRR SHHS 변수 목록](https://sleepdata.org/datasets/shhs/variables), EDF 채널 정의: [SHHS1 몽타주·샘플링 정보](https://sleepdata.org/datasets/shhs/pages/10-montage-and-sampling-rate-information-shhs1.md)

### 모델 입력 피처 (16개)

| # | 피처 | 뜻 | 단위 | SHHS 출처 | 워치/앱 출처 | 병원 |
|---|---|---|---|---|---|---|
| 1 | `avgsat` | 수면 중 평균 SpO₂ | % | CSV | SpO₂ 1분 기록 평균 | A |
| 2 | `avg_hr` | 수면 중 평균 심박 | bpm | CSV (`savbnbh` 등 4개 열 평균) | 수면 단계(깸 제외) 구간의 심박 평균 | A |
| 3 | `odi_1min` | 1분 단위 3%p 이상 저하 사건 수 | 회/시간 | EDF SaO₂ → 1분 변환 | SpO₂ 1분 기록 | A |
| 4 | `pct_lt90` | 분당 최소 SpO₂ < 90%인 분의 비율 | % | EDF SaO₂ → 1분 변환 | SpO₂ 1분 기록 | A |
| 5 | `min_spo2` | 분당 최소 SpO₂의 하위 1% 값 | % | EDF SaO₂ → 1분 변환 | SpO₂ 1분 기록 | A |
| 6 | `desat_drop_1min` | 저하 사건당 평균 저하 폭 | %p | EDF SaO₂ → 1분 변환 | SpO₂ 1분 기록 | A |
| 7 | `desat_min_frac` | 저하 사건에 걸친 분의 비율 | % | EDF SaO₂ → 1분 변환 | SpO₂ 1분 기록 | A |
| 8 | `slpprdp` | 총수면시간 | 분 | CSV | 수면 단계 (얕은+깊은+렘) | B |
| 9 | `slpeffp` | 수면 효율 | % | CSV | 수면 세션 `efficiency` | B |
| 10 | `times34p` | 깊은 수면(3~4단계) 비율 | % | CSV | 깊은 수면 / 총수면 | B |
| 11 | `waso` | 잠든 뒤 깨어 있던 시간 | 분 | CSV | 첫 잠듦 이후 깸 단계 합계 | B |
| 12 | `age_s1` | 나이 | 세 | CSV | 프로필 생년월일 | C |
| 13 | `gender` | 성별 (1=남, 2=여) | — | CSV | 앱 입력 | C |
| 14 | `bmi_s1` | BMI | kg/m² | CSV | 프로필 키·체중 | C |
| 15 | `neck20` | 목둘레 | cm | CSV | 앱 입력 | C |
| 16 | `ess_s1` | Epworth 졸림 척도 점수 | 점 | CSV | 앱 설문 | C |

- 병원 역할: **A** 산소포화도·심박, **B** 수면검사실(라벨 보유), **C** 일반 진료(인구통계·설문)
- 3~7번 SpO₂ 피처는 SHHS와 워치에 **같은 함수**(`oximetry_features.py`)를 적용해 계산한다. 학습과 추론의 피처 정의를 맞추기 위해서다.

> **구현 상태**: 위 16개 구성은 확정됐지만 학습 코드(`dataset.py`, `mpc/prepare_hospital_data.py`)와 추론 코드는 아직 이전 14개 피처 기준이다. 이전 14개는 3~7번 대신 XML 주석에서 뽑은 `odi`, `mean_desat_duration`, `mean_desat_drop`을 썼다. 이 3개는 아래 "XML 저하 피처를 쓰지 않는 이유" 때문에 교체한다.

### SpO₂ 1분 피처를 만드는 방법

갤럭시 워치가 저장하는 SpO₂는 **1분마다 평균·최소·최대**뿐이다. 그래서 SHHS의 1Hz SaO₂도 같은 형식으로 바꾼 뒤 피처를 계산한다.

1. **수면 구간 자르기**: SHHS는 XML 수면 단계로 첫 잠듦부터 마지막 수면까지만 남긴다. 워치는 수면 세션 구간을 쓴다.
2. **1분 묶기**: 50~100% 밖의 샘플은 잡음으로 버리고, 유효 샘플이 30초 미만인 분은 무효 처리한다.
3. **저하 사건 찾기**: 직전 3개 유효 분의 평균 중 최대값을 기준선으로 둔다. 분 최소값이 기준선보다 3%p 이상 낮으면 사건으로 보고, 이어지는 분은 한 사건으로 합친다.
4. 유효 시간이 1시간 미만이면 계산하지 않는다.

`min_spo2`는 순간 잡음을 피하려고 최소값 대신 하위 1% 값을 쓴다. `pct_lt90`는 짧은 저하를 놓치지 않도록 분당 평균이 아니라 분당 최소를 기준으로 한다.

### XML 저하 피처를 쓰지 않는 이유

SHHS XML의 "SpO2 desaturation" 사건은 산소포화도만으로 찾은 사건이 아니다. 수면기사가 기류 신호로 판정한 **무호흡·저호흡에 딸린 저하만** 기록돼 있다. 확인 결과 저하 사건의 99.9%가 호흡 사건 바로 옆에 붙어 있었다. 라벨(AHI)도 같은 호흡 사건으로 계산하므로, 이 사건으로 만든 피처는 **라벨 누출**이다. 또 기류 센서가 없는 워치로는 재현할 수 없다. NSRR도 SHHS1의 저하 주석은 신뢰하기 어렵다며 원신호에서 다시 계산하라고 안내한다 ([NSRR 포럼](https://sleepdata.org/forum/shhs1-desaturation-events-missing)).

### SHHS와 워치 데이터의 차이

| 항목 | 차이 | 영향 |
|---|---|---|
| 나이 | SHHS는 39세 이상(평균 63세) | 젊은 사용자는 학습 범위 밖 |
| SpO₂ 측정 | 손가락 산소측정기 vs 손목 광센서 | 워치 SpO₂가 1~2%p 높게 나올 수 있음 (`avgsat`, `min_spo2`, `pct_lt90`에 영향) |
| SpO₂ 해상도 | EDF는 1초 단위, 워치는 1분 요약 | 1분 변환으로 맞춤 |
| 수면 단계 | 뇌파(EEG) 판정 vs 심박·움직임 추정 | `times34p` 일치도가 가장 낮고, `waso`는 워치에서 짧게 잡히는 경향 |
| 수동 입력 | `gender`, `neck20`, `ess_s1` | 앱에서 입력받아야 함 |

---

## 2. 모델

수직 분할 **로지스틱 회귀**다. 병원 i는 자기 피처 x_i에 대한 가중치 w_i만 갖는다.

```
logit = w_A·x_A + w_B·x_B + w_C·x_C + b        (b는 한 병원만 보유)
P(AHI ≥ 15) = σ(logit)
```

피처는 각 병원이 자기 데이터로 표준화(StandardScaler)한다. 이 평균·표준편차는 추론에서도 그대로 쓴다.

### 2.1 학습 — SecretFlow SPU (MPC)

구현: `mpc/train_spu_party.py` (WSL2, `venv_wsl`)

```
[병원 i, 로컬]  자기 파일만 읽음 → 표준화 → logit_share_i = w_i · x_i
        │  logit_share_i, 라벨 y(병원 B), DP 노이즈 조각 n_i 를 SPU에 입력
        ▼
[SPU, ABY3 3자 MPC]  z = Σ logit_share_i
                     p̂ = σ̂(clip(z, −4, 4)),   σ̂(z) = 0.5 + 0.197z − 0.004z³
                     grad_z = (p̂ − y) / batch + Σ n_i
        │  grad_z 는 병원 i 에게만 공개
        ▼
[병원 i, 로컬]  ∂L/∂w_i = x_iᵀ · grad_z  →  Adam 업데이트
```

- **손실 `bce_clip_z_dp`**: BCE의 로짓 그래디언트 `σ(z) − y`에서 σ만 3차 다항식 σ̂로 바꿨다. MPC는 진짜 시그모이드(지수 함수)를 싸게 계산하지 못하기 때문이다. σ̂는 |z| ≈ 4.05 밖에서 감소하므로 z를 [−4, 4]로 먼저 자른다.
- **DP 노이즈**: 세 병원이 각자 N(0, σ²/3)을 뽑아 SPU에 넣는다. 합쳐진 노이즈는 SPU 안에서 그래디언트에 더해진 뒤 공개된다. 그래서 어느 병원도 노이즈 합을 혼자 알 수 없다.
- **역할**: 라벨은 병원 B(`hospital1`), bias는 병원 A(`hospital0`)가 갖는다 (`mpc/cluster_local.json`).
- **분할**: 8:2, `random_state=42` (test 1,072명). 정확도는 SPU 안에서 계산해 스칼라만 공개한다.
- **하이퍼파라미터** (현재 체크포인트): 100 epoch, batch 256, Adam lr 1e-2, DP σ = 0.01, seed 0

실행 (병원별 터미널 3개, 인자는 모두 같아야 함):

```bash
python mpc/prepare_hospital_data.py                       # 병원별 파일 생성 (최초 1회)
python -m mpc.train_spu_party --party hospital0 --epochs 100
python -m mpc.train_spu_party --party hospital1 --epochs 100
python -m mpc.train_spu_party --party hospital2 --epochs 100
python -m mpc.export_heaan2                               # → mpc/checkpoints/heaan2_model.txt
```

### 2.2 추론 — HEaaN2 CKKS

구현: `inference_v5/DistributedInferenceV5.cpp` (C++, HEaaN2)

```
환자 기기:  키 생성 (F16 프리셋, 슬롯 2¹⁵) → 회전키(1, 2, 4, 8)를 각 병원에 등록
            피처 14개 표준화 → 슬롯에 채움 → 암호화 → 같은 암호문을 세 병원에 전송
      ▼
병원 h:     enc(x) ⊙ mask_h        (mask_h: 자기 피처 위치에만 w_h, 나머지 0 — 평문)
            rescale → 회전 1, 2, 4, 8 로 슬롯 합산 → enc(w_h·x_h) 를 슬롯 0에 모음
      ▼
환자 기기:  enc(w_A·x_A) + enc(w_B·x_B) + enc(w_C·x_C)
            → 복호화 → + bias → σ(logit) → 임계값 비교
```

- 병원은 비밀키 없이 암호문×평문 곱셈, 회전, 덧셈만 한다.
- 추론의 시그모이드는 근사가 아니라 **정확한 σ**다. 복호화 뒤 평문으로 계산하므로 비용이 없다. 학습 때의 근사 σ̂와 판정은 같다: z > 0 ⇔ σ > 0.5 ⇔ σ̂ > 0.5.
- 모델 파일: `client_model.txt`(피처 이름, 스케일러, bias — 환자 기기용), `hospital{0,1,2}_model.txt`(병원별 인덱스·가중치). `mpc/export_heaan2.py`가 만든 `heaan2_model.txt`에서 나눠 만든다.
- 임계값: 기본 0.5. `inference_v5`는 민감도 0.85를 목표로 0.4를 쓴다 (`v5_1_threshold_optimizer.py`).

---

## 3. 성능

### 현재 배포 모델 (14개 피처, SPU 학습)

test 1,072명 (양성 43.4%):

| 항목 | 값 |
|---|---|
| 정확도 (임계값 0.5) | **0.891** |
| AUC | **0.960** |
| CKKS vs 평문 logit 최대 오차 | 2.0 × 10⁻⁶ |
| 환자 1명 추론 시간 | 592 ± 12 ms |
| 보정 오차 (ECE, 10구간) | 0.029 |

> 이 모델은 누출이 있는 XML 저하 피처를 쓰므로, **0.891을 워치 성능으로 주장할 수 없다.**

> `inference_v5/v5_metrics.csv`의 값(정확도 0.594, AUC 0.726)은 학습 체크포인트와 **다른 모델 파일**로 계산한 결과다. 같은 1,072명에 체크포인트 가중치를 쓰면 0.891 / 0.960이 나온다. CKKS 계산 자체는 정확하다(오차 2×10⁻⁶). 모델 파일을 `heaan2_model.txt`에서 다시 만들어야 한다.

### 피처 구성별 비교 (평문 로지스틱 회귀, 5겹 교차검증)

SHHS 4,715명 (EDF를 받은 인원 중 CSV 변수가 모두 있는 사람, 양성 44.1%):

| 피처 구성 | 정확도 | AUC | 민감도 | 특이도 | 비고 |
|---|---|---|---|---|---|
| 14개 (XML 저하 피처) | 0.893 | 0.965 | 0.866 | 0.915 | 누출 있음 |
| 11개 + CSV `ndes3ph`·`pctsa90h`·`minsat` | 0.884 | 0.953 | 0.837 | 0.921 | 초 단위, 워치로 재현 불가 |
| **16개 (확정 구성)** | **0.822** | **0.903** | 0.778 | 0.857 | **워치와 같은 1분 해상도** |
| 11개 (저하 피처 없음) | 0.702 | 0.773 | 0.624 | 0.765 | |

- **0.822가 워치에서 기대할 수 있는 성능에 가장 가까운 추정치다.** CSV 피처와의 차이(약 6%p)가 1분 해상도로 잃는 정보량이다.
- 16개 모델에서 가장 큰 가중치는 `desat_min_frac`(+1.45)와 `odi_1min`(+1.12)이다.
- SPU 학습(근사 σ̂ + DP 노이즈)은 평문보다 약간 낮을 수 있다. 이전 모델에서는 평문 0.897, SPU 0.891이었다.
- 16개 구성의 SPU 학습과 CKKS 추론은 아직 하지 않았다.

### 워치 데이터 분포 (사용자 1명, 155밤)

| 피처 | SHHS 평균 | 워치 중앙값 | SHHS 백분위 |
|---|---|---|---|
| `age_s1` | 63.0 | 23 | 0% |
| `avgsat` | 94.6 | 98.6 | 99.7% |
| `min_spo2` | 87.1 | 95.0 | 99.3% |
| `odi_1min` | 3.3 | 0.9 | 4.3% |
| `desat_min_frac` | 25.5 | 3.0 | 6.2% |

SHHS 중 50세 미만·AHI < 5·BMI 23~29인 사람(118명)도 평균 SpO₂가 96.3%다. 워치 값 98.6%는 나이와 건강 상태만으로는 다 설명되지 않아, 손목 센서가 높게 읽는 차이가 섞인 것으로 보인다.

---

## 4. 보안 고려사항

### 보호되는 것

| 대상 | 학습 | 추론 |
|---|---|---|
| 병원 원본 피처 | 각 병원 로컬에서만 사용 | 환자 기기에서 암호화 후 전송 |
| 라벨 (AHI) | 병원 B가 SPU에 비밀분산으로 입력 | 해당 없음 |
| 로짓, 예측값, 손실 | SPU 안에서만 계산 | 암호문, 환자만 복호화 |
| 병원별 그래디언트 | 노이즈를 더한 뒤 해당 병원에만 공개 | 해당 없음 |
| 병원별 가중치 | 각 병원 안에만 있음 | 평문 마스크로 각 병원이 보유 |
| 비밀키 | 해당 없음 | 환자 기기만 보유 |

### 공개되는 것

- 학습: 행 개수 n, 배치 인덱스(공개 시드), test 정확도, 배치 손실(`--reveal-loss`일 때만)
- 추론: 병원별 부분 logit 암호문 크기, 요청 횟수와 시점

### 전제와 한계

- **반정직(semi-honest) 모델**: ABY3는 3자 중 최대 1자가 프로토콜은 따르되 정보를 엿보는 상황을 가정한다. 악의적으로 프로토콜을 어기는 참여자나 둘 이상의 공모는 막지 못한다.
- **환자 정렬(PSI) 미구현**: 세 병원 파일이 같은 nsrrid 순서로 정렬돼 있다고 가정한다. 실제로는 PSI(Private Set Intersection)로 공통 환자를 맞춰야 한다.
- **단일 머신 시뮬레이션**: 세 병원을 한 머신의 프로세스로 실행했다. 다중 머신 배포는 검증하지 않았다.
- **DP 보장 미산정**: σ = 0.01 노이즈를 더하지만 (ε, δ) 프라이버시 예산은 계산하지 않았다. 형식적인 DP 보장이 아니라 그래디언트로 라벨을 추론하기 어렵게 만드는 장치로 봐야 한다.
- **추론 시 모델 노출**: 환자는 병원별 부분 logit을 각각 복호화한다. 많은 질의를 보내면 병원별 가중치를 역산할 수 있다. 병원 쪽에서 세 결과를 합쳐 보내거나 결과에 마스킹을 하면 줄일 수 있다.
- **암호문 전체 전송**: 환자는 14개 피처가 모두 담긴 같은 암호문을 세 병원에 보낸다. CKKS의 의미론적 안전성 덕분에 병원은 내용을 알 수 없지만, 병원별로 필요한 피처만 보내는 것보다 노출 범위가 넓다.
- **개인정보 파일**: `1024/`의 워치 내보내기에는 전화번호, 생년월일 등이 들어 있다. `.env`(NSRR 토큰)와 함께 저장소에 올리지 않는다.

---

## 5. 참고 논문

### 데이터
- Quan SF et al. *The Sleep Heart Health Study: design, rationale, and methods.* Sleep, 1997.
- Zhang GQ et al. *The National Sleep Research Resource: Towards a sleep data commons.* JAMIA, 25(10):1351–1358, 2018.

### 수직 연합학습 (VFL)
- Yang Q, Liu Y, Chen T, Tong Y. *Federated machine learning: Concept and applications.* ACM TIST, 10(2), 2019.
- Yang S, Ren B, Zhou X, Liu L. *Parallel distributed logistic regression for vertical federated learning without third-party coordinator.* arXiv:1911.09824, 2019.
- Hardy S et al. *Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption.* arXiv:1711.10677, 2017.
- Wu Y, Xing N, Chen G, Dinh TTA, Luo Z, Ooi BC, Xiao X, Zhang M. *Falcon: A privacy-preserving and interpretable vertical federated learning system.* PVLDB, 16(10):2471–2484, 2023.
- Liu Y, Kang Y, Zou T et al. *Vertical federated learning: Concepts, advances, and challenges.* IEEE TKDE, 2024. (arXiv:2211.12814)
- Chen D, Qu H, Xu G. *AegisFL: Efficient and flexible privacy-preserving Byzantine-robust cross-silo federated learning.* ICML, PMLR 235, 2024.

### VFL 추론 공격
- Fu C et al. *Label inference attacks against vertical federated learning.* USENIX Security, pp. 1397–1414, 2022.
- Li O et al. *Label leakage and protection in two-party split learning.* ICLR, 2022.
- Luo X, Wu Y, Xiao X, Ooi BC. *Feature inference attack on model predictions in vertical federated learning.* ICDE, pp. 181–192, 2021.

### MPC·동형암호·차분 프라이버시
- Mohassel P, Rindal P. *ABY3: A mixed protocol framework for machine learning.* ACM CCS, pp. 35–52, 2018.
- Ma J et al. *SecretFlow-SPU: A performant and user-friendly framework for privacy-preserving machine learning.* USENIX ATC, 2023.
- Cheon JH, Kim A, Kim M, Song Y. *Homomorphic encryption for arithmetic of approximate numbers.* ASIACRYPT, pp. 409–437, 2017.
- Abadi M et al. *Deep learning with differential privacy.* ACM CCS, 2016.

### 수면무호흡 예측 (ECG·SpO₂ 원신호, 임상 정보)
- Wang T et al. *Sleep apnea detection from a single-lead ECG signal with automatic feature-extraction through a modified LeNet-5 convolutional neural network.* PeerJ, 7:e7731, 2019.
- Chang H-Y, Yeh C-Y, Lee C-T, Lin C-C. *A sleep apnea detection system based on a one-dimensional deep convolution neural network model using single-lead electrocardiogram.* Sensors, 20(15):4157, 2020.
- Mortazavi E, Tarvirdizadeh B, Alipour K, Ghamari M. *Deep learning approaches for assessing pediatric sleep apnea severity through SpO2 signals.* Scientific Reports, 14:22696, 2024.
- Levy J, Álvarez D, Del Campo F, Behar JA. *Deep learning for obstructive sleep apnea diagnosis based on single channel oximetry.* Nature Communications, 14:4881, 2023.
- Nnamdi MC et al. *KindSleep: Knowledge-informed diagnosis of obstructive sleep apnea from oximetry.* arXiv:2603.04755, 2026.
- Retamales G et al. *Towards automatic home-based sleep apnea estimation using deep learning.* npj Digital Medicine, 7:144, 2024.

### 갤럭시 워치 (규제 문서·제품 자료)
- U.S. FDA. *De Novo classification request for Sleep Apnea Feature (DEN230041),* Samsung Electronics.
- U.S. FDA. *510(k) K201168: ECG Monitor App,* Samsung Electronics, 2020.
- Samsung Electronics. *Samsung Health Monitor ECG App Instructions for Use.*

### 웨어러블·심박 기반 수면무호흡 선별
- Guilleminault C et al. *Cyclical variation of the heart rate in sleep apnoea syndrome.* Lancet, 1984.
- Hayano J et al. *Screening for obstructive sleep apnea by cyclic variation of heart rate.* Circulation: Arrhythmia and Electrophysiology, 2011.
- Hayano J et al. *Quantitative detection of sleep apnea with wearable watch device.* PLoS ONE, 2020. ([PMC7652322](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7652322/))
- Papini GB et al. *Wearable monitoring of sleep-disordered breathing: estimation of the apnea–hypopnea index using wrist-worn reflective photoplethysmography.* Scientific Reports, 2020. ([PMC7421543](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7421543/))
- PhysioNet/Computing in Cardiology Challenge 2000: *Detecting Sleep Apnea from the ECG.* ([링크](https://archive.physionet.org/challenge/2000/))
