# Sub-model 설계 노트

## 1. 현재 구현 — `ClientSubModel` (`model.py:17-29`)

```python
class ClientSubModel(nn.Module):
    """Linear -> CubicAct -> Linear"""
    def __init__(self, input_dim, emb_dim=16):
        self.linear = nn.Linear(input_dim, emb_dim)
        self.act    = PolyActivation()      # 0.197x + 0.004x^3
        self.embed  = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.embed(self.act(self.linear(x)))
```

병원별로 독립된 인스턴스이며, 원본 feature는 이 모델 밖으로 절대 나가지 않는다.

**왜 이렇게 단순한가 — HE(CKKS) 곱셈 깊이 제약 때문.**
추론 시 이 sub-model은 CKKS 암호문 위에서 그대로 실행된다 (`model.py:138-161`,
`compute_sub_emb_he`). CKKS는 bootstrapping 없이는 곱셈을 할 때마다 ciphertext의
"level"이 소모되고, level이 바닥나면 더 이상 연산이 불가능하다. 지금 구조는:

| 연산 | HE 레벨 소모 |
|---|---|
| `linear` (행렬곱) | 0 (스칼라 곱, 레벨 소모 없음) |
| `z² = z*z` (CubicAct 1단계) | -1 |
| `z*(0.197+0.004z²)` (CubicAct 2단계) | -1 |
| `embed` (행렬곱) | 0 |

**총 2 레벨**만 쓴다. 이게 지금 sub-model이 "Linear→활성화→Linear" 2단으로
고정된 이유이며, bootstrapping 없이 안정적으로 돌아가는 예산이다.

## 2. FT-Transformer 도입 검토

### 2.1 왜 그대로 못 쓰는가

FT-Transformer는 학습된 그대로 CKKS 암호문에 넣을 수 없다. 세 곳이 막힌다:

- **Softmax (Attention)** — `exp()`와 나눗셈은 CKKS의 기본 연산(덧셈·곱셈)이 아님.
  다항식 근사가 필수.
- **LayerNorm** — 평균/분산 계산 후 `1/sqrt(var)` (역제곱근)도 비산술 연산.
- **곱셈 깊이 소진** — 레이어를 하나만 추가해도 지금 예산(2레벨)을 넘기 쉽다.
  bootstrapping 없이는 깊은 구조가 애초에 불가능.

### 2.2 병원별 feature 개수 문제 (`simulate.py:12-16`)

```
병원 A: SpO2, avg HR                        (2개)
병원 B: total sleep, efficiency, deep ratio (3개)
병원 C: age, sex, BMI                       (3개)
```

FT-Transformer의 핵심 강점은 **수십 개 컬럼 사이의 attention 상호작용**이다.
토큰이 2~3개뿐이면 attention이 학습할 수 있는 관계 자체가 적어서, 지금의
단순 MLP 대비 이점이 크지 않을 가능성이 높다. **HE 대응 작업에 들어가기 전에
평문 상태에서 먼저 정확도 이득이 실제로 있는지 검증 필요.**

### 2.3 그래도 도입한다면 — 현실적인 배치

1. **HE-호환 변형으로 처음부터 학습** (학습 따로 + 추론 시 사후 변환 아님):
   - Softmax → 다항식 근사 (SLOTHE 방식)
   - LayerNorm → 실행 시 배치 통계로 `rsqrt` 계산하는 대신, 학습 시점에
     고정된 평균/분산을 주변 선형층에 접어넣는(fold) 방식
   - FFN 활성화 → 기존 `PolyActivation` 재사용
2. **딱 1개 attention block만.** 표준 FT-Transformer처럼 블록을 여러 개
   쌓으면 곱셈 깊이 예산을 순식간에 초과한다.
3. Top-model 경계(§3)는 건드리지 않음 — sub-model 내부에서만 대체.

### 2.4 예상 리스크 (정직하게)

- softmax 다항식 근사는 활성화 함수 근사보다 훨씬 까다롭다 (음수 방지 +
  합=1 정규화까지 만족해야 함 — 저차 다항식은 학습 분포 밖에서 쉽게 발산).
- LayerNorm을 고정 통계로 fold하면 원래의 샘플별 적응적 정규화 이점이
  사라져 학습 안정성이 오히려 떨어질 수 있다.
- 위 두 근사 비용 + 토큰 수가 적어 얻는 이득이 작다는 점이 겹쳐, **HE 예산만
  더 쓰고 정확도는 지금 MLP와 비슷하거나 더 나쁠 위험**이 있다.

## 3. 전체 학습 모델 구조

### 3.1 구성 요소

```
병원 i (i = 0, 1, 2)
├─ 원본 feature X_i           ─ 병원 밖으로 절대 안 나감
├─ ClientSubModel (sub_i)     ─ 로컬, 평문 PyTorch
│    x_i → emb_i  (emb_dim=16)
└─ top-model 컬럼 슬라이스 W_top_i  ─ 전체 병원이 공유하는 top-model의 일부

공유 top-model: nn.Linear(emb_dim * 3, 1)   ← 반드시 "선형 1개 층"
```

### 3.2 Forward — block-linearity

```
logit_share_i = emb_i @ W_top_i.T (+ bias, 병원 0만)
logit         = sum_i(logit_share_i)          ← W_top 전체를 곱한 것과 수학적으로 동일
pred          = sigmoid_approx(logit)          ≈ 0.5 + 0.197·logit - 0.004·logit³
```

`logit_share_i`는 **다른 병원 데이터 없이 병원 i 혼자 계산 가능** (block-linearity).
이게 top-model이 "선형 1개 층"이어야 하는 이유다 — 층을 하나라도 더 넣으면
이 분해가 깨져서 top-model 전체를 MPC 안에서 돌려야 한다.

### 3.3 보안이 필요한 구간

```
[로컬, 평문]                    [MPC/HE 보호 필요]                  [로컬, 평문]
sub_i(X_i) → emb_i  →  logit_share_i = emb_i @ W_top_i  →  sigmoid_approx(sum) + loss  →  dL/d(logit_share_i) 만 병원 i에게 reveal → emb_i, W_top_i 로컬 backward
```

- **학습 (현재)**: `secret_sharing.py`의 `BeaverProvider` — additive secret
  sharing + Beaver Triple을 손수 구현. semi-honest 가정.
- **학습 (진행 중)**: `top_layer_jax.py` — 같은 수식을 순수 JAX로 재작성.
  SecretFlow의 SPU device에 태우면, `*` 연산이 자동으로 안전한 곱셈 프로토콜로
  컴파일된다 (Beaver Triple을 손으로 안 짜도 됨). WSL2 + `secretflow` 설치 진행 중.
- **추론**: `model.py`의 `compute_sub_emb_he` / `compute_logit_share_he` —
  TenSEAL CKKS로 병원별 sub-model + top-layer 슬라이스를 암호문 위에서 실행.
  최종 합산·복호화는 client(환자)가 로컬에서 수행.

### 3.4 요약 표

| 구간 | 실행 위치 | 보호 필요 여부 |
|---|---|---|
| 원본 feature → sub-model forward | 병원 로컬 | 불필요 (밖으로 안 나감) |
| `logit_share_i = emb_i @ W_top_i` | 병원 로컬 | 불필요 (block-linearity) |
| `sum(logit_share) → sigmoid → loss` | MPC(Beaver Triple 또는 SPU) | **필요** |
| `dL/d(logit_share_i)` reveal 이후 | 병원 로컬 (병원 i에게만) | 불필요 |
| 추론 시 sub-model + top-layer 슬라이스 | CKKS 암호문 | HE로 이미 보호됨 |
