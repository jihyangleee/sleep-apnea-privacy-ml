# CrypTen 기반 Mesh 학습 설계 (entry-point 제거안)

> **상태: 설계 논의 단계 — 아직 코드에 구현되지 않음.**
> 이 문서는 현재 `hospital_app.py` / `simulate.py`의 entry-point(hub) 구조를 완전히 없애고,
> 병원 3곳이 서로 직접 통신하는 진짜 mesh 구조로 바꾸기 위해 CrypTen 도입을 검토한 내용을 정리한다.

---

## 1. 현재 구조의 한계

- `simulate.py` (in-process 시뮬레이션): 3개 병원의 forward/backward 전체가 한 프로세스 안에서
  단일 PyTorch autograd 그래프로 실행된다. optimizer도 하나(`torch.optim.Adam`)가 전체 파라미터를
  담당한다.
- `hospital_app.py`의 `/train_distributed` (실제 HTTP 분산): 요청을 받은 병원이 그 세션 동안
  "entry-point"가 되어 나머지 두 병원과 star(hub-and-spoke) 형태로 통신한다.
  - peer의 `/train_forward`는 **평문 임베딩**을 그대로 반환한다 (`hospital_app.py:335`).
  - entry-point는 label(y)도 CSV 전체 로드를 통해 이미 평문으로 갖고 있다.
  - 즉 entry-point 역할이 매 세션 회전(rotate)하긴 하지만, 그 세션 동안은 사실상 신뢰된
    중앙 좌표자(trusted coordinator)이며, 임베딩·라벨이 여기서 전부 평문 노출된다.

### 검토했다가 기각한 대안: "diff / logit² 만 reveal"

Beaver triple은 top-layer sigmoid 하나에만 쓰이므로, backward에서 `diff = pred - y`와
`logit²`(forward의 triple1 결과 재사용)만 all-to-all로 reveal하면 새 Beaver triple 없이
각 병원이 로컬로 gradient를 계산할 수 있을 것처럼 보였다.

**그러나 이 방식은 안전하지 않다.** `sigmoid_approx(x) = 0.5 + 0.197x - 0.004x³`는
`f(-x) = 1 - f(x)` 관계를 가지므로, `logit²`만 알아도 `logit`은 `±L` 두 후보로 좁혀지고
`pred`도 두 후보(`p`, `1-p`)로 좁혀진다. 여기에 이미 공개된 `diff`를 대입해 두 가설
(`y = pred_+ - diff`, `y = pred_- - diff`)을 검증하면, `y ∈ {0, 1}`이라는 **이산 제약** 때문에
둘 중 하나만 유효한 값이 나와 **라벨이 그대로 복원된다.**

예: `logit=2, y=1` → 가설(+): `y=1.0` (유효) / 가설(-): `y≈0.276` (무효) → 부호 확정, `y` 노출.

→ 결론: "필요한 지점만 골라서 reveal"하는 손수 설계는 이산 라벨 제약 때문에 생각보다
쉽게 뚫린다. 이걸 막으려면 중간값을 아무것도 reveal하지 않는 진짜 MPC(secure backward)가
필요하다 → **CrypTen 검토로 이어짐.**

---

## 2. CrypTen이란

Meta(구 Facebook AI Research)의 PyTorch 기반 오픈소스 MPC(다자간 보안 연산) 라이브러리.

- HE(동형암호)가 아니라 **이 프로젝트의 `secret_sharing.py`와 같은 계열** — additive secret
  sharing + Beaver triple. 다만 sigmoid 하나가 아니라 **모든 텐서 연산**(add, matmul, mul,
  비교연산 등)을 이 방식으로 일반화해 지원한다.
- `MPCTensor`로 감싼 텐서는 여러 party 프로세스에 share로 나뉘어 존재하며, 모든 연산이
  프로세스 간 실제 통신(=mesh)을 동반하는 **능동적 다자간 프로토콜**로 실행된다. "암호문을
  한 곳에서 계산"하는 HE 추론과는 근본적으로 다른 모델이다.
- **autograd가 secret-shared 상태로 끝까지 유지된다.** `.backward()`를 호출해도 중간값이
  reveal되지 않으므로, 위 1절에서 발견한 diff/logit² 유출 문제가 구조적으로 발생하지 않는다.
- Beaver triple을 만드는 "dealer" 문제도 선택지가 있다: 기본은 우리 코드처럼 단순화된
  trusted-third-party 방식이지만, **OT(Oblivious Transfer) 기반 provider**를 쓰면 중앙 dealer
  없이 party들이 직접 triple을 생성할 수 있다.
- **미검증 사항**: 3-party 세팅에서 OT 기반 provider의 실제 지원 수준, 최신 PyTorch와의
  호환성, 유지보수 활성도는 아직 확인하지 않았다. 도입 전 확인 필요.

---

## 3. 적용 범위 — sub-model은 로컬, top-layer만 CrypTen

전체 모델을 CrypTen으로 감싸면 원래 secret sharing이 전혀 필요 없던 부분(각 병원의 sub-model,
`local_emb`)까지 MPC 오버헤드를 물게 되어 불필요하게 무거워진다. 대신 지금 코드가 이미
분리해 놓은 경계를 그대로 활용한다.

| 구간 | 실행 방식 | 이유 |
|---|---|---|
| Sub-model (`local_emb`) forward/backward | 병원 로컬, 평문 PyTorch (변경 없음) | 원본 feature가 애초에 병원 밖으로 안 나가므로 secret sharing이 필요 없음 |
| `logit_share_i = W_top_i @ emb_i (+bias)` | CrypTen (`MPCTensor`) | block-linearity로 통신은 없지만, 이후 sigmoid에서 secret 상태를 유지하려면 여기서부터 share여야 함 |
| Sigmoid 근사 (`0.5+0.197x-0.004x³`) | CrypTen | 기존 `BeaverProvider.sigmoid_approx`와 동일한 연산을 MPCTensor 연산으로 대체 |
| Loss(MSE) + backward | CrypTen | diff, gradient 모두 secret 상태 유지 — reveal 없음 |
| `dL/d(emb_i)`, `dL/d(W_top_i)` 경계 reveal | **병원 i에게만** party-scoped reveal | 아래 4절 참고 |

---

## 4. Forward / Backward 흐름

### Forward
1. 각 병원이 `local_emb`(로컬, 평문) → `logit_share_i` 계산까지는 지금과 동일 (통신 없음).
2. `logit_share_i`를 `MPCTensor`로 올려 CrypTen 연산 영역에 진입.
3. CrypTen이 `Σ logit_share_i`를 secret 상태로 유지한 채 sigmoid 다항식 근사를 계산
   (내부적으로 forward의 Beaver triple 곱셈 두 번 — 기존 triple1/triple2와 동일한 개념).
4. label도 각자 secret share로 들고 있다가 CrypTen 상에서 `diff`, `loss`까지 계산 — 이 시점에
   아무도 `logit`, `pred`, `diff`를 평문으로 보지 못함.

### Backward
1. `loss.backward()`가 CrypTen `MPCTensor` 그래프 위에서 실행되며, forward와 동일한 이유로
   등장하는 모든 secret×secret 곱셈(예: sigmoid 도함수 계산)에 Beaver triple을 재사용.
2. top-layer 경계에서 `dL/d(emb_i)`, `dL/d(W_top_i)`가 share 상태로 나온다.
3. **여기서 구조적으로 한 번은 평문 reveal이 필요하다** — 병원 i의 sub-model이 평문
   PyTorch라서, share가 아닌 실제 텐서 값으로 `.backward()`를 이어가야 하기 때문이다.
4. 단, CrypTen은 이 reveal을 **병원 i 자신에게만** 하고 다른 두 병원에게는 노출하지 않는
   party-scoped reveal을 지원한다. 이는 1절의 실패한 방식(전원에게 reveal)과 결정적으로 다르며,
   최종적으로 노출되는 값도 `diff`·`logit²`처럼 따로따로가 아니라 `dL/d(logit) = 2·diff·sigmoid'(logit)`
   하나로 합쳐진 스칼라라서 미지수 대비 관측치 수가 줄어든다(다만 이 조합에 대한 엄밀한
   무누출 증명은 별도로 하지 않았음 — CrypTen을 쓰는 이유가 바로 이런 미묘한 지점을 직접
   증명하지 않고도 안전하게 넘기기 위함).
5. 병원 i는 revealed된 `dL/d(emb_i)`로 자기 sub-model에 대해 로컬 backward + 로컬 optimizer
   step을 수행. `dL/d(W_top_i)`도 마찬가지로 병원 i가 자신의 top-layer 슬라이스를 로컬
   optimizer로 갱신.

### CrypTen 필요 여부 정리

"top-layer만 CrypTen"(3절)은 **forward만이 아니라, 그 top-layer에 대응하는 backward 전체**
(loss → sigmoid 미분 → logit 합의 gradient)까지 포함한다. sub-model 방향으로는 forward·backward
어느 쪽도 확장되지 않는다.

| 구간 | CrypTen 필요 | 이유 |
|---|---|---|
| sub-model forward | ✗ | 원본 feature가 병원 밖으로 안 나가므로 애초에 secret sharing 불필요 |
| `logit_share_i = W_top_i @ emb_i` (forward) | ✓ | 이후 sigmoid에서 secret 상태를 유지하려면 여기서부터 share여야 함 |
| sigmoid 근사, loss (forward) | ✓ | `logit`이 세 병원을 다 더한 joint 값이라 아무도 평문으로 못 봐야 함 |
| `loss.backward()` → sigmoid 미분 → logit 합의 gradient (backward) | ✓ | `sigmoid'(x) = 0.197 - 0.012x²`에 `x²` 항이 또 있어, forward와 마찬가지로 secret×secret 곱셈(Beaver triple)이 backward에도 등장 |
| `dL/d(emb_i)`, `dL/d(W_top_i)` reveal 시점 이후 | ✗ | party-scoped reveal로 병원 i에게만 평문 복원, 그 뒤는 순수 로컬 PyTorch |
| sub-model backward + optimizer.step() | ✗ | 병원 i 로컬 처리 — 지금 코드(`hospital_app.py:346-349`)의 `/train_backward`와 동일한 패턴 |

**주의**: backward 일부만 hand-roll로 빼려 하면 1절의 실패 사례(diff·logit²만 all-to-all reveal)로
되돌아간다. sigmoid 미분 계산 중 나오는 중간값(`x²`, `diff` 등)을 어설프게 평문으로 꺼내는 순간
라벨 역산 문제가 재발하므로, top-layer 안의 모든 연산(forward + backward)이 하나도 빠짐없이
CrypTen 안에서 끝나야 하고, sub-model 경계를 넘는 순간(reveal)에만 평문으로 나온다.

---

## 5. 통신 비용 비교

- **naive하게 전체 모델을 CrypTen으로 감싸는 경우**: sub-model 내부 연산까지 전부 MPC를 타서
  지금 방식보다 훨씬 무거움 — 비권장.
- **top-layer만 CrypTen으로 스코프를 좁히는 경우**: forward는 지금 `BeaverProvider`와 통신량이
  비슷하고(triple 2개 + reveal), backward도 "제대로 안전하게" 만들려면 hand-rolled 구현이든
  CrypTen이든 결국 비슷한 양의 Beaver triple 연산이 필요하다 (1절에서 확인했듯 "reveal 몇 개로
  가볍게 끝내기"는 안전하지 않으므로, 안전한 버전끼리 비교하면 비용 차이가 크지 않음).
- 즉 CrypTen 도입의 가치는 "더 가벼워서"가 아니라 **"직접 짜면 놓치기 쉬운 leak을 검증된
  라이브러리로 방지"**하는 데 있다.

---

## 6. 남은 확인 사항 (TODO)

- [ ] CrypTen의 3-party(world_size=3) 지원 수준 및 예제 확인
- [ ] OT 기반 triple provider의 실제 동작/성숙도 확인 (dealer 완전 제거 가능한지)
- [ ] 현재 설치된 PyTorch 버전과의 호환성 확인, 최근 유지보수 활성도 확인
- [ ] `crypten.nn`으로 top-layer(선형 결합 + sigmoid 근사)를 어떻게 표현할지 프로토타입
- [ ] party-scoped reveal API(`reveal_to` 등) 정확한 사용법 확인
- [ ] `hospital_app.py`의 FastAPI/httpx 통신 계층을 CrypTen 자체 communicator와 어떻게
      공존시킬지 (완전 대체 vs 병행 실행) 결정
