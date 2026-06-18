# Sleep Apnea VFL + HE

수면 무호흡증 예측을 위한 **Vertical Federated Learning + 동형암호화(CKKS) 추론** 시스템.

3개 병원이 원시 데이터를 공유하지 않고 협력 학습하며, 환자 기기(Galaxy Watch)는
모든 추론 과정에서 데이터를 암호화한 채로 예측 결과만 받는다.

---

## 시스템 아키텍처 및 파이프라인 (Architecture & Pipeline)

본 프로젝트는 수직 분할 학습(Vertical Federated Learning, VFL) 환경에서 환자의 프라이버시를
완벽하게 보장하며 수면 무호흡증을 예측하는 다층 방어 아키텍처를 가진다. 전체 파이프라인은
마스킹 단계를 제외하고 다음과 같이 **학습(Training)**과 **추론(Inference)** 과정으로 나누어
진행된다.

---

### 1. 학습 과정 (Training Process)

학습 단계에서는 **평문(Plaintext) 도메인**에서 로컬 연산을 수행하되, 병원 간 데이터 전송 시
**차분 프라이버시(DP)**와 **덧셈 비밀분할(Additive Secret Sharing)**을 결합하여 가중치를
안전하게 동시 업데이트한다.

**단계 1. 평문 도메인에서의 로컬 임베딩 계산**
- 각 병원(A, B, C)은 스마트워치에서 수집된 환자의 생체 데이터(HR, SpO2, 수면 시간)를 입력받아
  로컬 GPU에서 독립적으로 임베딩을 계산한다.
- 특징 추출 백본인 **LKCNN + 1DSE + BiGRU** 구조를 통과하며, 암호화 도메인 호환성을 위해
  비선형 활성화 함수는 **3차 다항식 근사 ReLU**를 적용한다.

```
f(x) = 0.197x + 0.004x³
```

**단계 2. 차분 프라이버시(DP) 노이즈 추가**
- 병원 간 공모 공격 및 통계적 역추적 추론을 방어하기 위해, 추출된 로컬 임베딩 벡터에
  라플라스 노이즈(`Laplace(0, Δ/ε)`)를 선제적으로 주입한다.

**단계 3. 덧셈 비밀분할 (Additive Secret Sharing)**
- 노이즈가 섞인 임베딩 벡터를 수학적으로 3등분하여 조각(Share)으로 쪼갠다.

```
embedding_A = share_A1 + share_A2 + share_A3
```

- 각 병원은 자신의 조각을 다른 병원들과 교환하여, 어떤 단일 병원도 다른 병원의 원본 임베딩
  분포를 알 수 없도록 격리한다.
  - **병원 A 보유 조각:** (`share_A1, share_B1, share_C1`)
  - **병원 B 보유 조각:** (`share_A2, share_B2, share_C2`)
  - **병원 C 보유 조각:** (`share_A3, share_B3, share_C3`)

**단계 4. 분할 탑 모델 선형 결합 (Split Top Model)**
- 각 병원은 수집한 조각들을 결합(Concat)한 후, 탑 모델의 가중치 행렬과 곱하는 선형 연산을
  수행한다. 이 과정은 순수 선형 결합이므로 병원 간 별도의 통신이 발생하지 않는다.

```
logit_share_A = concat([share_A1, share_B1, share_C1]) · W_top_model + b_top_model
```

**단계 5. 최종 오차 산출 및 역전파 (Beaver Triples)**
- 비밀분할 상태에서 안전하게 오차를 계산하기 위해 **Beaver Triples** 알고리즘을 도입하여
  곱셈 연산을 수행한다.
- 최종 예측 확률값은 **3차 다항식 근사 시그모이드(Sigmoid)** 함수를 거쳐 산출되며, 계산된
  손실(Loss)을 바탕으로 탑 모델과 서브 모델의 가중치를 동시에 역전파하여 업데이트한다.

```
σ(x) ≈ 0.5 + 0.197x - 0.004x³
```

---

### 2. 추론 과정 (Inference Process)

추론 단계에서는 환자의 데이터 기밀성을 극대화하기 위해 **CKKS 동형암호(Homomorphic
Encryption)**를 사용한다. 학습된 모델 가중치(`W, b`)는 고정된 평문 상수로 사용되며, 환자의
데이터는 전 과정에서 암호화된 상태로 연산된다.

**단계 1. 환자 디바이스 도메인 암호화**
- 환자의 스마트워치에서 수집된 생체 데이터는 외부로 전송되기 전, 환자 본인의 디바이스에서
  자체 공개키(`pk_patient`)를 사용하여 CKKS 암호문 상태로 변환된다.

```
enc_features = Encrypt(watch_features, pk_patient)
```

**단계 2. 암호화 상태의 서브모델 연산**
- 암호화된 데이터(`enc_features`)가 각 병원으로 전달되면, 병원들은 앞서 학습이 완료된 고정
  가중치를 상수 플레인텍스트로 취급하여 암호문과 곱해주는 선형 연산을 수행한다. 이 과정에서
  환자의 평문 데이터는 병원에 절대 노출되지 않는다.

```
enc_embedding_A = apply_submodel_A_encrypted(enc_features)
```

**단계 3. 암호화 상태의 탑 모델 결합**
- 각 병원의 서브모델이 출력한 암호화된 임베딩들을 한데 모아 탑 모델의 가중치를 행렬 곱
  연산으로 결합하여 하나의 암호화된 최종 로짓(`enc_logit`)을 생성한다. 동형암호의 특성 덕분에
  암호화된 상태 그대로 결합 연산이 완료된다.

```
enc_logit = enc_embedding_A · W_top_A + enc_embedding_B · W_top_B + enc_embedding_C · W_top_C
```

**단계 4. 환자 복호화 및 최종 예측**
- 최종 연산된 암호문 결과(`enc_logit`)를 환자의 디바이스로 안전하게 반환한다.
- 환자는 오직 자신만 보유하고 있는 개인 비밀키(`sk_patient`)로 복호화를 수행한 후, 시그모이드
  함수를 적용하여 최종 수면 무호흡증 위험도 확률(0~1 사이의 값)을 최종 확인한다.

```
logit_final = Decrypt(enc_logit, sk_patient)
risk_score  = sigmoid(logit_final)
```

---

## 학습 vs 추론 구조 비교

| | 학습 | 추론 |
|---|---|---|
| 연산 도메인 | 평문 (로컬 GPU) | CKKS 암호문 |
| 보호 대상 | **임베딩 데이터** (3-way 덧셈 비밀분할 + Laplace DP noise) | **환자 raw feature** (CKKS 암호화) |
| 모델 가중치 | 학습 중 탑/서브 모델 동시 역전파 업데이트 | 고정된 평문 상수로 취급, 암호문과 곱해짐 |
| 비선형 처리 | Beaver Triple로 보호된 share 상태에서 다항식 근사 시그모이드 적용 | ciphertext 그대로 다항식 근사 ReLU 적용 (-2 HE 레벨) |
| 결합 연산 | 각 병원이 share를 concat 후 `W_top`과 선형 결합 (통신 불필요) | 각 병원의 enc(embedding)에 `W_top` 행렬곱 후 합산 |
| 최종 출력 | 탑 모델 결합 → 시그모이드 → BCE Loss | 환자 기기에서 decrypt → 시그모이드 → 위험도 확률 |
| 키 보유 | 해당 없음 (비밀분할로 격리) | 환자만 CKKS 비밀키(`sk_patient`) 보유 |

---

## 모델 구조

### Sub-model (병원별 private)
```
LKCNN + 1DSE + BiGRU → 3차 다항식 근사 ReLU (PolyAct)
```
각 병원의 feature만 처리. 다른 병원과 공유되지 않음.

### Top-model (병원 간 공유, 동일 가중치)

```
W_top : Linear(total_emb_dim → 1)     # total_emb_dim = emb_dim × 3 (기본 16×3=48)
b_top : (1,)
```

학습 시 3개 병원 모두 **동일한 `W_top`, `b_top` 파라미터**를 공유하며 동시에 역전파한다.
순수 선형이라 비밀분할(SS)된 share 위에서 곧바로 분산 계산이 가능하다.

```
concat_share_j = [share_A_j, share_B_j, share_C_j]            (batch, 3·emb_dim)
logit_share_j  = concat_share_j @ W_top.weight.T + (b_top  if j==0 else 0)
logit          = Σ_j logit_share_j
               = (Σ_j concat_share_j) @ W_top.weight.T + b_top
               = concat(emb_A, emb_B, emb_C) @ W_top.weight.T + b_top      ← 선형성으로 등호 성립
```

- `Σ_j share_j = emb` 이므로 share에 먼저 곱하고 나중에 더해도 결과가 같다 → **곱셈이 끼지 않아 Beaver Triple이 필요 없다** (`model.py: HospitalModel.logit_share`).
- 바이어스 중복 합산을 막기 위해 `j==0`인 병원만 `b_top`을 더한다 (`is_first` 플래그).
- 곱셈(비선형)이 필요한 곳은 로짓을 합산한 **이후의 시그모이드 근사 단계뿐**이며, 거기서만 Beaver Triple 2회(`x²`, `x³`)가 들어간다 (`secret_sharing.py: BeaverProvider.sigmoid_approx`).

추론(CKKS) 시에는 같은 `W_top`을 **컬럼 단위로 슬라이스**해서 각 병원이 자기 임베딩에 대응하는 부분만 들고 있는다 (`model.py: HospitalModel.build_he_weights`):

```
W_top_A = W_top.weight[:, 0:16]      # 병원 A 보유 — 자기 emb_A 컬럼만
W_top_B = W_top.weight[:, 16:32]     # 병원 B 보유
W_top_C = W_top.weight[:, 32:48]     # 병원 C 보유

enc_logit_i = enc_embedding_i @ W_top_i.T          (ciphertext × plaintext, -1 HE 레벨)
enc_logit   = Σ_i enc_logit_i + b_top              ← coordinator가 합산 후 1회만 더함
```

- 어떤 병원도 `W_top` 전체를 복원할 필요가 없다 — 자기 컬럼 슬라이스만으로 충분하다.
- 이 곱셈 1회가 CKKS 파라미터 표의 "sub-model 선형 3" 중 마지막 1단계에 해당한다.

### PolyAct: `f(x) = 0.197x + 0.004x³`
- CKKS 동형암호 호환 (다항식)
- `x * (0.197 + 0.004*x²)` 형태로 인수분해 → ciphertext 곱셈 2회 (-2 HE 레벨)
- ReLU 대신 사용

### Sigmoid 근사: `σ(x) ≈ 0.5 + 0.197x - 0.004x³`
- 학습 시 Beaver Triple 2회(x², x³)로 비밀분할 상태에서 계산
- BCELoss 입력 직전에 적용되어 어떤 병원도 평문 로짓을 보지 않음

### CKKS 파라미터
| 파라미터 | 값 |
|---|---|
| `poly_modulus_degree` | 16384 |
| `coeff_mod_bit_sizes` | [60, 40×7, 60] |
| `global_scale` | 2⁴⁰ |
| 가용 곱셈 레벨 | 7 |
| 소비 레벨 | 5 (sub-model 선형 3 + CubicAct 2) |
| 남은 레벨 | 2 (헤드룸) |

---

## 데이터셋

**Sleep Heart Health Study (SHHS-1)** — NSRR

| feature | SHHS 변수명 | 담당 병원 |
|---|---|---|
| SpO₂ 평균 산소포화도 | `avgsao2` | 병원 A |
| 평균 심박수 | `avg_hr` | 병원 A |
| 총수면시간 (분) | `slptime` | 병원 B |
| 수면 효율 (%) | `slp_eff` | 병원 B |
| 깊은수면 비율 (%) | `timest34p` | 병원 B |
| 나이 | `age_s1` | 병원 C |
| 성별 | `gender` | 병원 C |
| BMI | `bmi_s1` | 병원 C |

레이블: `ahi_a0h3a >= 15` → 중등도 이상 수면무호흡

대안 데이터: DREAMT v2.1.0 (`--dreamt`), 기본값: 합성 더미 데이터

---

## 프라이버시 보장 범위

| 항목 | 학습 | 추론 |
|---|---|---|
| 환자 raw feature | - | 환자 디바이스에서 CKKS 암호화 후 전송 |
| 임베딩 노출 | 3-way 덧셈 비밀분할 + Laplace DP noise | ciphertext 상태로만 교환 |
| 병원 간 공모 / 통계적 역추적 | DP noise로 차단 | ciphertext이므로 해당 없음 |
| 로짓(logit) 노출 | Beaver Triple → 아무도 평문 미열람 | ciphertext, 환자만 복호화 가능 |
| 모델 가중치 `W, b` | 병원들이 동일 가중치 보유, 공동 역전파 | 고정 평문 상수로 사용 (가중치 자체는 비밀 아님) |
| 최종 예측 확률 | BCE Loss 계산용으로 share 합산 후 노출 | 환자 디바이스에서만 decrypt + sigmoid |

---

## 파일 구조

```
├── dataset.py         데이터 로드 (SHHS / DREAMT / 합성)
├── model.py           HospitalModel (학습+추론 통합), ServerTopModel, PolyActivation
├── secret_sharing.py  additive_split, apply_dp_noise, BeaverProvider
├── simulate.py        run_distributed_simulation (신규), run_vertical_simulation (레거시)
├── he_client.py       build_he_context, HospitalHE, HEInference (benchmark 호환)
├── hospital_app.py    FastAPI 병원 서버 (병원별 독립 실행)
├── watch_app.py       Galaxy Watch 역할 웹 클라이언트 (다른 컴퓨터에서 실행)
├── client_gui.py      Galaxy Watch 역할 데스크톱 GUI 클라이언트 (Tkinter)
├── schemas.py         FastAPI Pydantic 요청/응답 모델
├── client.py          VerticalClient (레거시 학습)
├── server.py          VerticalFLServer (레거시 학습)
├── benchmark.py       HE vs 평문 / SS overhead 벤치마크
└── main.py            진입점 (--mode vertical / distributed / he-infer)
```

---

## 실행

```bash
pip install torch tenseal numpy pandas scikit-learn
```

### 학습

```bash
# 분산 학습 (SS + DP noise + rotating coordinator)
python main.py --mode distributed

# SHHS 데이터
python main.py --mode distributed --csv shhs1-dataset-0.21.0.csv

# DREAMT 데이터
python main.py --mode distributed --dreamt physionet.org/files/dreamt/2.1.0

# DP noise 조절 (기본 0.01, 0이면 비활성화)
python main.py --mode distributed --dp-sigma 0.005

# 레거시 학습 (semi-honest server 단일 코디네이터)
python main.py --mode vertical
```

### Watch 클라이언트 웹앱 (다른 컴퓨터에서 실행)

```bash
# Watch 컴퓨터에서
COORDINATOR_URL=http://<병원서버IP>:8001 uvicorn watch_app:app --host 0.0.0.0 --port 9000
```

브라우저에서 `http://localhost:9000` 접속 → 수치 입력 → 암호화 후 예측 요청

```
[Watch 컴퓨터 :9000]               [병원 컴퓨터]
  feature 입력 폼                   :8001  hospital_app
  CKKS 암호화 (secret key 보유)     :8002  hospital_app
  POST /infer ──────────────────▶   :8003  hospital_app
  enc_logit 수신
  복호화 → 확률 표시
```

### FastAPI 서버 (병원별 독립 실행)

```bash
pip install fastapi uvicorn httpx

# 터미널 3개에서 각각 실행
HOSPITAL_ID=0 uvicorn hospital_app:app --port 8001
HOSPITAL_ID=1 uvicorn hospital_app:app --port 8002
HOSPITAL_ID=2 uvicorn hospital_app:app --port 8003
```

```bash
# 체크포인트 없을 때 — 서버 실행 후 학습 트리거
curl -X POST http://localhost:8001/train

# Galaxy Watch 추론 요청 (coordinator 병원 선택)
# Watch: POST /infer with {enc_xi_b64: {"0": ..., "1": ..., "2": ...}}
# 응답: {enc_logit_b64: "..."} → Watch가 복호화 → 수면무호흡 확률

# 헬스 체크
curl http://localhost:8001/health
```

**추론 내부 흐름 (3라운드 병원간 HTTP):**
```
Watch → POST /infer (coordinator)
  Round 1: coordinator → 각 병원 /compute_emb   (enc_xi → enc_emb)
  Round 2: coordinator → 각 병원 /compute_h_share (enc_emb_all → enc_h_share)
           coordinator: Σenc_h_share → PolyAct → enc_h_act
  Round 3: coordinator → 각 병원 /compute_logit_share (enc_h_act → enc_logit_share)
           coordinator: Σenc_logit_share = enc_logit → Watch
```

### HE 추론 데모 (시뮬레이션)

```bash
python main.py --mode he-infer
```

```
[HE Inference] checkpoint mode: distributed

Profile 0: 젊은 남성 (정상)
  Plaintext prob  : 0.2341
  HE prob         : 0.2342  |err|=0.000134  (entry=Hospital B)
```

### 벤치마크

```bash
python benchmark.py --n-infer 10 --n-plain 500 --n-batches 30
```
