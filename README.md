# Sleep Apnea VFL + HE

수면 무호흡증 예측을 위한 **수직 연합학습(Vertical FL) + 동형암호(CKKS) 추론** 시스템.

3개 병원이 원시 데이터를 서로 공개하지 않고 협력 학습하며, 환자 기기(Galaxy Watch 역할)는
추론 과정에서 데이터를 암호화한 채로 예측 결과만 받는다.

---

## 현재 상태 한눈에 보기

| 영역 | 구현 상태 |
|---|---|
| 모델 | **수직 로지스틱 회귀**(기본, `--model linear`) / sub-model MLP + linear top(`--model mlp`) |
| 피처 | 15개 (SHHS 기본 12 + SpO₂ desaturation 3), 3개 병원에 분배 |
| 학습 | 병원별 로컬 평문 연산 + MPC로 시그모이드·손실·그래디언트 계산 + 그래디언트 DP 노이즈. 손실은 MSE + 3차 다항식 시그모이드. MPC는 두 경로: **자체 Beaver 시뮬레이션**(`main.py`, Windows) / **SecretFlow SPU**(`mpc/`, WSL2) |
| MPC 프레임워크 | SecretFlow SPU(ABY3, 3자) — **logit 합산·시그모이드·MSE·그래디언트**를 SPU에서 실행. 라벨은 무작위 덧셈 비밀분할, 그래디언트는 병원별로만 공개(+DP). 서브모델/선형 가중치 곱은 각 병원 로컬. 체크포인트를 저장해 `he-infer`로 이어진다 |
| 추론 | TenSEAL CKKS. linear 모델은 병원당 암호문×평문 행렬곱 1회, 활성화 없음 |
| 서버/GUI | `app/hospital_app.py`는 MLP 체크포인트만 지원, `app/client_gui.py`는 8피처 기준(미갱신) |
| 정확도 | 평문 기준 정확도 약 0.89 / AUC 약 0.96 (아래 실험 결과). MPC용 다항식 시그모이드로 학습하면 자체 시뮬레이션·SPU 모두 약 0.83~0.85 |


---

## 시스템 아키텍처

### 1. 학습 (Training)

구현: `simulate.py: run_distributed_simulation` (`main.py --mode distributed`).

```
[병원 i, 로컬 평문]  자기 피처 x_i  ──►  logit_share_i = w_i · x_i      (linear)
                                        (mlp: sub-model → emb_i → W_top 열 조각 → logit_share_i)
        │
        ▼  logit_share_i 는 로짓의 덧셈 share (블록 선형성) — 다른 병원의 피처·임베딩은 오가지 않는다
[Beaver Triple 시뮬레이션]  pred = σ̃(Σ logit_share_i)      σ̃(x) = 0.5 + 0.197x − 0.004x³   (곱셈 2회)
        │
        ▼  라벨 y 도 덧셈 비밀분할(additive_split)
  loss = mean((pred − y)²)      ← MSE: 그래디언트가 뺄셈뿐이라 비밀분할과 잘 맞는다 (BCE는 나눗셈 필요)
        │
        ▼  역전파 (임베딩/로짓 그래디언트에 Gaussian DP 노이즈 σ 추가)
[병원 i, 로컬]  자기 가중치 갱신
```

- **로컬 연산**: 서브모델/선형 가중치 곱은 각 병원이 자기 피처만으로 계산한다. 원본 피처와
  임베딩은 병원 밖으로 나가지 않는다.
- **Beaver Triple**: 로짓 합산 이후의 시그모이드 근사(x², x³)에서만 필요하다
  (`secret_sharing.py: BeaverProvider`). 세 병원을 한 프로세스에서 흉내 내는 **시뮬레이션**이며
  실제 네트워크 통신은 없다. 이 경로에서는 시그모이드까지만 share 상태로 계산하고, 손실은 share를
  합쳐 복원한 `pred − y`를 평문에서 제곱해 구한다.
- **DP 노이즈**: 로컬 출력에 걸린 그래디언트에 Gaussian(σ=`--dp-sigma`, 기본 0.01)을 더해
  그래디언트로부터의 라벨 추론을 막는다.

#### SPU 경로 (`mpc/simulate_spu.py`, WSL2)

```
[병원 i, 로컬 평문]  logit_share_i = w_i · x_i           (mlp: sub-model → emb_i → W_top 열 조각)
        │  logit_share_i, 라벨의 무작위 덧셈 share(additive_split) 를 .to(spu) 로 SPU에 넣는다
        ▼
[SecretFlow SPU, ABY3 3자 MPC]   pred = σ̃(Σ logit_share_i)  →  pred − y  →  MSE  →  ∂loss/∂logit_share_i
        │  손실(스칼라)은 모니터링용으로 공개, 그래디언트 i 는 병원 i 에게만 공개(.to(pyu_i))
        ▼
[병원 i, 로컬]  그래디언트 + Gaussian DP 노이즈 → 역전파 → 가중치 갱신
```

- 자체 시뮬레이션과 달리 **`pred − y`와 손실 제곱도 SPU 안에서** 계산되어 중간값이 어느 병원에도 보이지 않는다.
- 라벨 y는 어느 병원도 통째로(또는 상수배로) 갖지 않는다.
- 매 배치의 모양을 같게 하려고 마지막 불완전 배치는 버린다(SPU 프로그램을 한 번만 컴파일). 첫 배치에
  ray·SPU 시작과 컴파일 비용(약 2~3분, `/mnt/d` 기준)이 들고 이후에는 배치(256)당 약 0.4초다.
- 세 병원은 한 머신의 ray 프로세스로 시뮬레이션한 것이며, `sf.reveal`은 그 단일 드라이버로 값을
  가져온다. 실제 다중 머신 배포는 검증하지 않았다.
- 서브모델·선형 가중치 곱은 SPU에 올리지 않고 로컬 평문으로 남는다.

### 2. 추론 (Inference) — CKKS

구현: `main.py --mode he-infer`, `model.py: LinearHospitalModel / HospitalModel`.

```
환자(Watch): 병원별 피처 슬라이스를 각각 CKKS로 암호화 (비밀키는 환자만 보유)
      │  enc(x_A) ─► 병원 A     enc(x_B) ─► 병원 B     enc(x_C) ─► 병원 C
      ▼
각 병원: enc(logit_i) = enc(x_i) · w_i          (linear: 암호문 × 평문 행렬곱 1회, 활성화 없음)
         (mlp: enc(x_i) → Linear → CubicAct → Linear → W_top 열 조각, 약 5 레벨)
      ▼
합산: enc(logit) = Σ enc(logit_i) + b      (b 는 한 번만 더함)
      ▼
환자: Decrypt → sigmoid → 위험도 확률
```

- 병원은 비밀키 없이 암호문 연산(Evaluator)만 수행한다. 평문 피처는 병원에 노출되지 않는다.
- `he-infer` 데모에서는 무작위로 뽑은 진입 병원이 세 병원의 `enc(logit_i)`를 합산한다.
  `app/client_gui.py` 경로에서는 환자가 직접 합산한다.
- 5개 더미 프로파일에서 평문 확률과 CKKS 확률의 차이(|err|)는 **linear 모델이 6자리 소수까지 0**,
  **mlp 모델은 최대 약 0.01**이었다(3차 활성화의 CKKS 근사 오차가 더해진다).

### CKKS 파라미터 (`he_client.py`)

| 파라미터 | 값 |
|---|---|
| `poly_modulus_degree` | 16384 |
| `coeff_mod_bit_sizes` | [60, 40×7, 60] |
| `global_scale` | 2⁴⁰ |
| 가용 곱셈 레벨 | 7 |
| MLP 경로 소비 레벨 | 약 5 (sub-model 선형 3 + CubicAct 2) — 추정, 미실측 |
| linear 경로 소비 레벨 | 평문 행렬곱 1회 — 실제 레벨 소모는 미실측 |

TenSEAL(SEAL 기반)은 CKKS 부트스트래핑을 제공하지 않는다. 레벨이 7을 넘는 구조가 필요하면
파라미터 확대(`poly_modulus_degree` 32768) 또는 다른 라이브러리(HEaaN, OpenFHE 등)가 필요하다.

---

## 모델 구조

### linear (기본): `LinearHospitalModel`

```
병원 i:  logit_share_i = Linear(n_i → 1, bias 없음)(x_i)
합산:    logit = Σ_i logit_share_i + b        (b 는 병원 0이 보유)
확률:    σ(logit)
```

파라미터는 피처당 가중치 하나와 바이어스 하나뿐(15 + 1개)이다. 활성화가 없어 CKKS·MPC 모두에서
가장 싸다.

### mlp (선택): `HospitalModel`

```
sub-model (병원별 private): Linear(n_i → 16) → PolyAct → Linear(16 → 16)  → emb_i
top-model (공유):           Linear(48 → 1)     # 병원별 열 조각 W_top_i 로 분해
logit = Σ_i W_top_i · emb_i + b_top
```

PolyAct: `f(x) = 0.197x + 0.004x³` (ReLU의 다항식 근사, `x·(0.197 + 0.004x²)`로 인수분해해 CKKS
곱셈 2회). 입력이 작을 때는 선형 항이 지배적이라 비선형성이 약하고, 입력 범위를 벗어나면 x³ 항이
폭주하므로 clipping·정규화가 필요하다.

### 시그모이드 근사

`σ̃(x) = 0.5 + 0.197x − 0.004x³`. |x|≈4에서 최대가 되고 그 뒤로 **감소**한다(비단조).
학습 정확도 저하의 원인이다 (아래 "MPC용 손실 근사의 대가").

---

## 데이터셋

**Sleep Heart Health Study (SHHS-1, NSRR)** — CSV(`shhs/datasets/shhs1-dataset-0.21.0.csv`) +
이벤트 주석 XML(`shhs/polysomnography/annotations-events-nsrr/shhs1/`, 2,691개).

모델 입력은 `dataset.MODEL_FEATURES` 순서의 **15개**다.

| 병원 | 피처 (SHHS 변수) | 인덱스 |
|---|---|---|
| **A** 심박·산소 | `avgsat`(평균 SpO₂), `avg_hr`(HR 4개 열의 평균), `odi`, `mean_desat_duration`, `mean_desat_drop` | 0, 1, 12, 13, 14 |
| **B** 수면검사실 | `slpprdp`(총 수면), `slpeffp`(수면 효율), `times34p`(깊은 수면 %), `waso`, `sleep_latency`(0/1 이진값) | 2, 3, 4, 8, 9 |
| **C** 병원/클리닉 | `age_s1`, `gender`, `bmi_s1`, `neck20`, `ess_s1` | 5, 6, 7, 10, 11 |

- 라벨: `ahi_a0h3a >= 15` (중등도 이상 수면무호흡). 로드된 2,444명 중 양성 47.1%.
- **desat 피처 3종**(`odi`, 이벤트 평균 지속시간, 평균 하락 폭)은 XML의 `SpO2 desaturation`
  이벤트에서 뽑는다(`extract_desat_features.py`). 라벨(AHI)을 그대로 재구성하게 되므로 호흡
  이벤트(무호흡·저호흡)는 일부러 쓰지 않았다. 파싱에 약 12초 걸려 결과를
  `shhs/datasets/desat_features_all.csv`에 캐시한다(지우면 다시 생성).
- XML은 `nsrrid` 200001~202700 범위만 받아져 있어 이 범위와 CSV가 겹치는 참가자만 쓴다.
- 결측 처리: `avg_hr`은 4개 HR 열의 (결측 무시) 평균이므로 원시 HR 열 결측으로 참가자를 버리지
  않는다. (예전 `df.dropna()`는 이 때문에 5.8천 명을 2.5천 명으로 줄였다.)
- `shhs/polysomnography/edfs/`의 원시 EDF(86개)는 **학습·추론에서 쓰지 않는다.**

---

## 실험 결과 (평문, 5개 시드 평균±표준편차)

`python -m experiments.architecture.compare_model_families` — 로지스틱 회귀 대비 은닉층이
가치가 있는지 본다. 시드마다 층화 분할이 다르다. MLP는 배포와 같은 3병원 분할·PolyAct를 쓰고
손실은 평문 BCE다.

| 모델 | 8피처 (n=2,660) 정확도 / AUC | 13피처 +desat (n=2,564) | **15피처 배포 (n=2,444)** |
|---|---|---|---|
| 로지스틱 회귀 | 0.713 / 0.781 | 0.889 / 0.961 | **0.891 / 0.960** |
| 2차 항 로지스틱 회귀 | 0.708 / 0.776 | 0.880 / 0.958 | 0.881 / 0.956 |
| MLP (`mlp`와 같은 구조) | 0.712 / 0.781 | 0.889 / 0.961 | 0.895 / 0.960 |
| MLP + 비선형 top | 0.713 / 0.782 | 0.888 / 0.961 | 0.891 / 0.960 |
| HistGradientBoosting (상한선, 튜닝 없음) | 0.680 / 0.741 | 0.894 / 0.963 | 0.893 / 0.963 |

읽는 법:

- **모든 구성에서 로지스틱 회귀 = MLP** (차이가 표준편차 안). 은닉층의 이득이 없다.
- **성능을 좌우하는 것은 모델이 아니라 desat 피처**다. desat 없는 8피처는 정확도 0.71,
  desat을 넣으면 0.89.
- 8피처 표의 GBDT가 낮은 것은 표본이 2~3천 명이고 튜닝하지 않아 과적합된 것으로 보이며,
  "상한선"으로 믿기 어렵다.
- 8피처와 13피처는 로더가 달라 표본 수가 다르다 (`feature_ablation_experiment.load_with_features`
  는 예전 `dropna` 방식을 유지).

### MPC용 손실 근사의 대가

15피처, 고정 분할 1개(`random_state=42`, 테스트 489명), 100 epoch, 선형 모델로 손실만 바꿔 비교했다
(인라인 진단으로 실행, 스크립트는 저장소에 없음).

| 학습 방식 | 정확도 | max\|logit\| |
|---|---|---|
| sklearn 로지스틱 회귀 | 0.894 | — |
| BCE + 진짜 sigmoid | 0.894 | 32.0 |
| MSE + 진짜 sigmoid | 0.898 | 28.4 |
| **MSE + 3차 다항식 시그모이드 (현재 MPC 방식)** | **0.841** | 8.9 |
| 그래디언트 `σ̃(z)−y` (나눗셈 없는 대안) | 0.69~0.72 (분할 3개) | 70~95 (발산) |
| **SPU 학습** (MSE + 3차 다항식, batch 256, lr 1e-2, DP σ=0.01, 300 epoch) | 0.82~0.85 (25 epoch마다 측정, 최고 0.847) | — |

#### 평문 vs MPC 학습 모델 (같은 분할, 정확도·AUC·민감도·특이도)

`python -m experiments.architecture.evaluate_checkpoints vertical_model.pt vertical_model_spu.pt` — 테스트 489명
(양성 233명, 47.6%), 학습에 쓰지 않은 분할. 괄호는 부트스트랩 95% 구간. AUC는 로짓 순위로, 민감도·특이도는
정확도와 같은 임계값(p>0.5)으로 계산했다.

| 모델 | 정확도 | AUC | 민감도 | 특이도 |
|---|---|---|---|---|
| 평문 sklearn 로지스틱 회귀 (BCE) | 0.894 (0.87–0.92) | 0.961 (0.95–0.97) | 0.893 (0.85–0.93) | 0.895 (0.85–0.93) |
| MPC 학습, 자체 시뮬레이션 (`vertical_model.pt`, 100 epoch) | 0.847 (0.82–0.88) | 0.929 (0.91–0.95) | 0.833 (0.79–0.88) | 0.859 (0.82–0.90) |
| MPC 학습, SPU (`vertical_model_spu.pt`, 300 epoch 마지막 가중치) | 0.832 (0.80–0.86) | 0.921 (0.90–0.94) | 0.785 (0.74–0.83) | 0.875 (0.83–0.91) |

- MPC 학습 모델은 평문보다 정확도가 약 5~6%p, AUC가 약 3~4%p 낮다. AUC도 떨어졌으므로 임계값 문제가 아니라
  **순위 자체가 나빠진 것**이다(시그모이드 근사로 로짓이 좁은 범위에 갇히기 때문).
- SPU 모델은 특히 **민감도가 0.785로 낮아** 양성의 약 21%를 놓친다(평문 약 11%). 선별 용도라면 임계값을 낮춰 보정해야 한다.
- 두 MPC 모델의 차이(자체 시뮬레이션 vs SPU)는 구간이 크게 겹쳐 유의하지 않다. 체크포인트마다 한 번만 학습했고
  SPU 모델은 최고 epoch가 아니라 마지막 epoch의 가중치다. 평문과의 차이에 대한 쌍체 검정은 하지 않았다.

3차 다항식이 |x|≈4 이후 감소해서 logit이 작은 범위에 갇히고, 대안 그래디언트는 발산했다.
`main.py --mode distributed`(자체 시뮬레이션)와 `mpc/simulate_spu.py`(SPU)로 실제 학습한 결과도
같은 수준(0.83~0.85)이라, 정확도 손실의 원인은 MPC 구현이 아니라 시그모이드 근사와 MSE 손실이다.
단조 근사(예: 1차)나 넓은 구간을 맞춘 고차 다항식은 아직 시험하지 않았다.

---

## 프라이버시 보장 범위

| 항목 | 학습 | 추론 |
|---|---|---|
| 환자 raw feature | 각 병원이 자기 피처만 로컬에서 사용 | 환자 디바이스에서 CKKS 암호화 후 전송 |
| 임베딩/부분 로짓 | 병원 밖으로 나가지 않음 (로짓은 덧셈 share) | 암호문으로만 교환 |
| 로짓(logit) 노출 | Beaver Triple 시뮬레이션 상태로 시그모이드 계산 | 암호문, 환자만 복호화 |
| 라벨 | 덧셈 비밀분할 (`additive_split`) | 해당 없음 |
| 그래디언트 기반 라벨 추론 | Gaussian DP 노이즈 (σ 기본 0.01) | 해당 없음 |
| 모델 가중치 | 병원이 자기 가중치 보유 | 고정 평문 상수 (가중치 자체는 비밀 아님) |

이 표는 **자체 시뮬레이션 기준**이다. SPU 경로는 손실 계산(`pred − y`, 제곱)까지 SPU 안에서 수행하고
라벨을 `additive_split`으로 나눠 넣는다. 그래디언트는 병원 i에게만 공개된다.

---

## 파일 구조

루트에는 여러 곳에서 import 되는 **공용 모듈**만 두고, 단독 실행되는 스크립트는 용도별 폴더로
묶는다. 폴더 안의 스크립트는 **저장소 루트에서 `python -m` 으로 실행**한다.

```
├── main.py                    진입점 (--mode distributed / he-infer, --model linear|mlp)
│
│   ── 공용 모듈 (루트) ──────────────────────────────────────────────
├── dataset.py                 SHHS 로드(CSV + desat XML), MODEL_FEATURES, 더미 워치 프로파일
├── model.py                   LinearHospitalModel, HospitalModel, ClientSubModel,
│                              PolyActivation, FeatureTokenizer
├── secret_sharing.py          additive_split, apply_dp_noise, BeaverProvider
├── simulate.py                run_distributed_simulation (SLEEP_FEATURE_GROUPS, model_type)
├── he_client.py               build_he_context (환자·병원 공유 CKKS 컨텍스트)
├── schemas.py                 FastAPI Pydantic 요청/응답 모델
├── extract_desat_features.py  SpO2 desaturation 피처 추출 (odi 등, NSRR XML)
├── linear_attention_jax.py    linear attention 1블록 (JAX, 평문 실험용)
├── linear_attention_deep_jax.py  다층 linear attention (pre-LN + residual)
├── top_layer_jax.py           SPU에 올리는 top-layer loss/grad
│
├── app/                       실제 서비스 (FastAPI + GUI)
│   ├── hospital_app.py        FastAPI 병원 서버 — MLP 체크포인트 전용
│   └── client_gui.py          Galaxy Watch 역할 데스크톱 GUI (Tkinter) — 8피처 기준, 미갱신
│
├── mpc/                       SecretFlow SPU (MPC) 실험 — WSL2 전용
│   ├── simulate_spu.py        SPU(ABY3, 3자) 학습 — linear/mlp, 체크포인트 저장 (`--out`)
│   └── spu_top_layer_test.py  top-layer만 SPU에 올린 최소 동작 테스트
│
├── experiments/               평문 실험 (HE/MPC 없이 정확도만 확인)
│   ├── simulate_linear_attention.py   단일 블록 베이스라인
│   ├── simulate_watch_domain_gap.py   PSG → 워치 도메인 갭 측정 (앞의 5개 피처만)
│   ├── architecture/          모델 구조·학습 안정화 비교
│   │   ├── compare_model_families.py     로지스틱 회귀 vs MLP vs GBDT (8/13/15피처)
│   │   ├── evaluate_checkpoints.py       평문 vs MPC 학습 체크포인트 (정확도·AUC·민감도·특이도, 부트스트랩 CI)
│   │   ├── compare_depth_experiment.py   깊이 (clipping 이전, 발산)
│   │   ├── compare_depth_clipped.py      깊이 (clipping + final LN)
│   │   ├── compare_width_clipping.py     너비 d, gradient clipping
│   │   ├── compare_normalization_waso.py 정규화 방식
│   │   ├── compare_loss_function.py      손실 함수
│   │   └── compare_mlp_vs_attention.py   MLP vs linear attention
│   └── features/              피처 구성 실험
│       ├── feature_ablation_experiment.py  피처 ablation (+ load_with_features 로더)
│       ├── combined_desat_experiment.py    8피처 + waso/latency + desat 피처
│       └── event_feature_experiment.py     이벤트 기반 피처
│
├── shhs/                      NSRR SHHS 데이터 (CSV, XML 주석, EDF는 미사용)
├── physionet.org/             DREAMT v2.1.0 (현재 파이프라인에서 미사용)
│
└── docs/
    ├── linear_attention.md       P2P VFL + SPU/CKKS + FT-Transformer 설계 문서 (구현과 차이 있음)
    ├── submodel.md               sub-model 설계 노트 (HE 곱셈 깊이 제약)
    └── CRYPTEN_MESH_DESIGN.md    CrypTen mesh 구조 설계 (미구현, 논의 단계)
```

`docs/linear_attention.md`는 FT-Transformer + Linear Attention을 SPU에서 학습하는 **목표 설계**이고,
현재 코드는 그 구조를 구현하지 않았다. Linear Attention은 평문 JAX 실험(`experiments/`)에서만
돈다.

### 제거된 실험 — CVHR (ECG 기반 심박 변동)

`extract_cvhr_feature.py` / `extract_cvhr_acat.py` / `simulate_ppg_cvhr.py` /
`combined_feature_experiment.py` 는 삭제했다. SHHS R-point 주석에서 뽑은 CVHR index는
AHI와 상관이 있었지만(r=0.474, n=89) ECG 수준의 피크 정밀도가 필요했고, PPG 노이즈를
씌운 시뮬레이션에서 재현되지 않아 워치에서 얻을 수 없는 피처로 판단해 드롭했다.
또한 rpoint CSV가 받아진 ~89명으로 학습셋이 줄어 정확도 측정 자체가 불가능했다.
코드가 필요하면 커밋 `1e042ec1` 이전에서 복구할 수 있다.

---

## 실행

### 개발 환경 (venv 2개)

secretflow가 Windows를 지원하지 않아 환경을 둘로 나눠 쓴다. 용도가 겹치지 않으니 둘 다 유지해야 한다.

| 환경 | 파이썬 | 용도 |
|---|---|---|
| `venv_win/` | 3.10 (Windows) | 기본 개발 환경 — `main.py`, `app/`, `experiments/` 전부 (jax 0.6.2 / torch / tenseal / fastapi) |
| `venv_wsl/` | 3.10 (WSL2) | `mpc/` 전용 — secretflow 1.13 + ray + jax 0.4.26 (구버전 jax 고정이라 분리) |

```bash
# Windows
venv_win\Scripts\activate
pip install -r requirements.txt

# WSL2 (SPU/MPC 실험에만 필요)
source venv_wsl/bin/activate
```

### 학습

`--csv`는 필수다 (합성 데이터 기본값과 `--dreamt` 옵션은 없다).

```bash
# 기본: 수직 로지스틱 회귀, 30 epoch
python main.py --mode distributed --csv shhs/datasets/shhs1-dataset-0.21.0.csv

# epoch 늘리기 (MSE + 3차 시그모이드 기준 100 epoch 부근에서 수렴)
python main.py --mode distributed --csv shhs/datasets/shhs1-dataset-0.21.0.csv --epochs 100

# sub-model MLP + linear top 구조
python main.py --mode distributed --csv shhs/datasets/shhs1-dataset-0.21.0.csv --model mlp

# DP noise 조절 (기본 0.01, 0이면 비활성화)
python main.py --mode distributed --csv shhs/datasets/shhs1-dataset-0.21.0.csv --dp-sigma 0.005
```

학습이 끝나면 `vertical_model.pt`에 저장되며 체크포인트에는 `model_type`이 들어간다.

#### SPU(MPC)로 학습 (WSL2)

```bash
wsl -d Ubuntu
cd /mnt/d/sleep_apnea && source venv_wsl/bin/activate
python -m mpc.simulate_spu --epochs 300 --batch-size 256 --lr 0.01 --out vertical_model_spu.pt
# 스모크 테스트: --epochs 2 --max-batches 2
```

첫 배치에 수 분의 시작·컴파일 비용이 든다. 저장한 체크포인트는 Windows에서 `main.py --mode he-infer --ckpt vertical_model_spu.pt`로
CKKS 추론에 쓴다.

### HE 추론 데모 (시뮬레이션)

```bash
python main.py --mode he-infer
```

```
[HE Inference] 워치 → 병원별 partial feature 암호화 (model_type=linear)

  Profile 0: 고위험 | 54세 남성 | 비만·저산소
    Plaintext prob  : 0.9868
    HE prob         : 0.9868  |err|=0.000000  (entry=Hospital C)
```

더미 워치 프로파일(`dataset.py`)은 15차원이며 값은 임의로 만든 것이다.

### 실험 스크립트 실행

```bash
# 저장소 루트에서 -m 으로 실행한다 (폴더 안에서 직접 실행하면 import가 깨진다)
python -m experiments.architecture.compare_model_families          # 8 / 13 / 15피처 전부
python -m experiments.architecture.compare_model_families 15feat   # 하나만
python -m experiments.architecture.compare_depth_clipped
python -m experiments.features.combined_desat_experiment
python -m mpc.simulate_spu                                          # WSL2
```

### Watch 클라이언트 (Tkinter 데스크톱 GUI, 코디네이터 없음)

병원 서버와는 별개의 컴퓨터(환자/Galaxy Watch 역할)에서 실행한다 — secret key를 가진 유일한 주체이므로
병원 서버와 같은 머신에 두면 안 된다. **현재 GUI는 8피처 기준이라 15피처 모델과 맞지 않는다.**

```bash
pip install requests tenseal
python -m app.client_gui
```

`app/client_gui.py` 상단의 `HOSPITAL_URLS`를 병원 서버 IP:포트로 맞춘 뒤 실행한다.

### FastAPI 서버 (병원별 독립 실행, MLP 구조 전용)

```bash
pip install fastapi uvicorn

# 터미널 3개에서 각각 실행
HOSPITAL_ID=0 uvicorn app.hospital_app:app --port 8001
HOSPITAL_ID=1 uvicorn app.hospital_app:app --port 8002
HOSPITAL_ID=2 uvicorn app.hospital_app:app --port 8003

# 체크포인트 없을 때 — 서버 실행 후 학습 트리거 (mlp 구조로 학습)
curl -X POST http://localhost:8001/train

# 헬스 체크
curl http://localhost:8001/health
```

서버가 읽는 체크포인트는 `--model mlp`로 만든 것이어야 한다. `--model linear`로 저장한
체크포인트는 이 서버에서 로드되지 않는다.
