# Sleep Apnea VFL + HE

수면 무호흡증 예측을 위한 **Vertical Federated Learning + 동형암호화(CKKS) 추론** 시스템.

3개 병원이 원시 데이터를 공유하지 않고 협력 학습하고,
환자 기기(Galaxy Watch)는 기능을 암호화한 채로 예측 결과를 받는다.

---

## 학습 — Vertical FL + Additive SS + DP Noise

```
병원 A (SpO₂, avg HR)       병원 B (수면지표 3개)        병원 C (나이, 성별, BMI)
      │ sub_model_A                 │ sub_model_B                 │ sub_model_C
      │ Linear → PolyAct            │ Linear → PolyAct            │ Linear → PolyAct
      ↓ emb_A                       ↓ emb_B                       ↓ emb_C
      │
      ├─── additive split + DP noise ──────────────────────────────────┐
      │                                                                 │
      │  병원 j 수신: concat_share_j = [share_A_j, share_B_j, share_C_j]  │
      │  h_share_j = concat_share_j @ W_top1 + b1/n   ← SS 선형 분산  │
      │                                                                 │
      │  [랜덤 coordinator 선정]                                        │
      │  h_linear = Σ h_share_j     ← 평문으로 봄 (semi-honest)       │
      │  h_act    = PolyAct(h_linear)   ← 병원 간 피처 상호작용       │
      │  logit    = h_act @ W_top2                                      │
      │  loss     = BCEWithLogitsLoss(logit, y)                        │
      └─────────────────────────────────────────────────────────────────┘

  ** Beaver Triple (PolyAct 적용)
  각 party j:  e_j = h_share_j - a_j  →  broadcast
               f_j = h_share_j - b_j  →  broadcast
  공개값:      e = h_linear - a  (a를 모르면 h_linear 복원 불가)
               f = h_linear - b
  결과:        act_share_j = c_j + f*a_j + e*b_j + [j=0]*e*f
               Σ act_share_j = PolyAct(h_linear)  ← h_linear 비공개 상태

  [Backward]
  coordinator → d_loss/d_logit (BCELoss gradient)
             → d_loss/d_act_share_j 브로드캐스트
  병원 j     → d_loss/d_W_top2 += act_share_j.T @ d_loss
             → d_loss/d_W_top1, d_loss/d_emb_i 역전파
  (시뮬레이션에서는 PyTorch autograd가 위 흐름을 자동 계산)

  ** Gradient DP noise (label 보호)
  d_loss/d_emb_i 가 bottom model로 흐르기 전에 Gaussian noise 주입.
  gradient의 부호/크기에서 label을 역추론하는 공격을 차단.
  (embedding 전송 시 DP noise와 동일한 sigma 사용)

프라이버시 메커니즘
  - Additive SS       : 개별 share는 정보이론적으로 랜덤 → 원본 embedding 복원 불가
  - DP noise (forward): 전송 전 Gaussian noise 주입 → 통계적 추론 차단
  - DP noise (backward): gradient에 Gaussian noise 주입 → label 역추론 차단
  - Beaver Triple     : h_linear 평문 재구성 없이 PolyAct 계산 → h_linear 비공개
  - Rotating coord    : 매 배치마다 logit 합산 coordinator 랜덤 선정 → 신뢰 분산
  - 노출 최소화       : coordinator가 보는 건 h_linear(32d) 아닌 logit(스칼라)뿐
```

---

## 추론 — CKKS 동형암호 + W_top Additive Share

```
                     [Galaxy Watch / 환자 기기]  (CKKS 비밀키 보유)
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │ enc([f0,f1])             │ enc([f2,f3,f4])          │ enc([f5,f6,f7])
         ↓                          ↓                           ↓
     병원 A                      병원 B                      병원 C
  sub_model_A (CKKS)          sub_model_B (CKKS)          sub_model_C (CKKS)
  Linear → PolyAct_CKKS       Linear → PolyAct_CKKS       Linear → PolyAct_CKKS
         │ enc(emb_A)                │ enc(emb_B)                 │ enc(emb_C)
         └──────────── enc(emb) 교환 (ciphertext, 복호화 불가) ───┘
                                    │
     각 병원 i:  enc(h_share_i) = Σⱼ enc(emb_j) @ W_top1_share_i[j]  + b1_share_i
                                    │
     [랜덤 coordinator]  enc(h_linear) = Σ enc(h_share_i)
                                    │
                    PolyAct_CKKS(enc(h_linear))   ← -1 HE 레벨
                                    │
     각 병원 i:  enc(logit_share_i) = enc(h_act) @ W_top2_share_i + b2_share_i
                                    │
     coordinator: enc(logit) = Σ enc(logit_share_i)
                                    │
                     [Galaxy Watch / 환자 기기]
                     sigmoid(decrypt(enc(logit))) → 수면무호흡 확률

프라이버시 메커니즘
  - 환자 feature  : 병원별 slice만 암호화 전송, 다른 병원 feature 접근 불가
  - enc(emb)      : ciphertext 교환, 어떤 병원도 복호화 불가
  - W_top1, W_top2: 추론 시 additive share 분산 보유, 누구도 전체 가중치 미보유
  - PolyAct       : 암호화 상태로 연산, coordinator도 h_linear 평문 미열람
  - 최종 logit    : 환자 기기만 복호화 가능
```

---

## 학습 vs 추론 구조 비교

| | 학습 | 추론 |
|---|---|---|
| SS 대상 | **데이터** (embedding additive split) | **모델 가중치** (W_top additive split) |
| 이유 | 평문 share는 선형 연산에서 분산 가능 | ciphertext는 분할 불가; 같은 enc에 weight share 적용 |
| 비선형 (PolyAct) | coordinator가 h_linear 보고 적용 | CKKS 다항식 — ciphertext 상태 그대로 |
| coordinator | 랜덤 선정, h_linear 평문 노출 | 랜덤 선정, ciphertext만 봄 |
| 최종 sigmoid | BCEWithLogitsLoss 내부 적용 | 환자 기기에서 decrypt 후 적용 |

---

## 모델 구조

### Sub-model (병원별 private)
```
Linear(|feature_i| → emb_dim) → PolyAct
```
각 병원의 feature만 처리. 다른 병원과 공유되지 않음.

### Top-model (병원 간 공유, 동일 가중치)
```
Linear1(total_emb → 32) → PolyAct → Linear2(32 → 1) → logit
```
- Linear1: SS로 분산 계산 (각 병원이 자기 share에 적용, 합산)
- PolyAct: 병원 간 cross-feature 비선형 상호작용 학습
- Linear2: 추론 시 additive share로 분산

### PolyAct: `f(x) = x * (x + 0.5) = x² + 0.5x`
- CKKS 동형암호 호환 (다항식)
- 인수분해 형태 → ciphertext 곱셈 1회 (-1 HE 레벨)
- ReLU / sigmoid 대신 사용

### CKKS 파라미터
| 파라미터 | 값 |
|---|---|
| `poly_modulus_degree` | 16384 |
| `coeff_mod_bit_sizes` | [60, 40, 40, 40, 40, 40, 60] |
| `global_scale` | 2⁴⁰ |
| 가용 곱셈 레벨 | 5 |
| 소비 레벨 | 2 (sub-model + top-model PolyAct) |
| 남은 레벨 | 3 |

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
| 환자 raw feature | - | 병원별 encrypted slice만 전달 ✅ |
| embedding 노출 | SS + DP noise (forward) ✅ | ciphertext 교환 ✅ |
| label → gradient 역추론 | DP noise (backward) ✅ | 해당 없음 |
| h_linear 노출 | Beaver Triple → 아무도 평문 미열람 ✅ | ciphertext, 누구도 복호화 불가 ✅ |
| 모델 가중치 W_top | 병원들이 동일 보유 | additive share 분산 보유 ✅ |
| 최종 logit | coordinator 스칼라 봄 ⚠️ | 환자만 복호화 ✅ |

> ⚠️ 학습 단계 한계: `h_linear`(32d)는 Beaver Triple로 보호되나,
> coordinator가 최종 `logit`(스칼라 1개)을 평문으로 봄. loss 계산에 필수적인 한계.

---

## 파일 구조

```
├── dataset.py         데이터 로드 (SHHS / DREAMT / 합성)
├── model.py           HospitalModel (학습+추론 통합), ServerTopModel, PolyActivation
├── secret_sharing.py  additive_split, apply_dp_noise, BeaverProvider
├── simulate.py        run_distributed_simulation (신규), run_vertical_simulation (레거시)
├── he_client.py       build_he_context, HospitalHE, HEInference (benchmark 호환)
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

### HE 추론 데모

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
