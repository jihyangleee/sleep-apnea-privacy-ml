# P2P 수직 연합학습 및 하이브리드 암호화 기반 FT-Transformer 시스템 아키텍처

본 문서는 코디네이터(중앙 서버) 없는 완전 분산(Peer-to-Peer) 환경에서 **SecretFlow SPU 기반 MPC(학습)**와 **클라이언트 키 기반 CKKS 동형암호(추론)**를 결합하여 수면무호흡증을 예측하는 프라이버시 보존 머신러닝(PPML) 시스템 아키텍처를 정의합니다.

---

## 1. 시스템 개요

* **목적:** 각 병원에 수직 분할(Vertical Partitioning)되어 있는 환자 생체 데이터를 타 기관에 노출하지 않고 공동 학습하고, 외부 클라이언트의 암호화된 요청을 수신하여 추론 수행
* **대상 데이터셋:** Sleep Heart Health Study (SHHS-1) 기반 8개 피처
* **예측 타겟 (Label):** `ahi_a0h3a >= 15` (중등도 이상 수면무호흡증 여부)
* **주요 특징:**
  * **Zero-Trust P2P 구조:** 중앙 코디네이터/서버 부재, 병원 간 비밀키 공유 없음
  * **하이브리드 암호화 파이프라인:** 학습 단계(MPC)와 추론 단계(CKKS FHE)의 이원화
  * **SecretFlow VFL 및 SPU 적용:** SecretFlow 프레임워크의 수직 연합학습(VFL) 모듈 및 SPU(Secure Processing Unit) 백엔드를 활용하여 안전한 분산 MPC 학습 수행
  * **클라이언트 중심 CKKS 추론:** 병원은 비밀키($SK$)를 보유하지 않고 암호문 연산(Evaluator)만 수행
  * **Linear Attention 적용:** Softmax 제거를 통해 MPC 통신 라운드 및 CKKS 연산 깊이(Depth) 최소화

---

## 2. 데이터 구성 및 분할 (Data Partitioning)

총 8개의 피처가 3개 병원에 수직으로 분할되어 분포합니다.

| 담당 기관 | SHHS 변수명 | 피처 설명 | 역할 |
| :--- | :--- | :--- | :--- |
| **병원 A** | `avgsao2`<br>`avg_hr` | SpO₂ 평균 산소포화도<br>평균 심박수 | Local Feature Holder |
| **병원 B** | `slptime`<br>`slp_eff`<br>`timest34p` | 총수면시간(분)<br>수면 효율(%)<br>깊은수면 비율(%) | Local Feature Holder |
| **병원 C** | `age_s1`<br>`gender`<br>`bmi_s1` | 나이<br>성별<br>BMI | Local Feature Holder + **Label Holder** |

---

## 3. 모델 아키텍처: FT-Transformer with Linear Attention

### 3.1 구조 설계
* **Bottom Model (Local Feature Tokenizer):**
  * 각 병원이 자사 피처(2~3개)를 $d$차원의 임베딩 벡터(토큰)로 변환하는 선형 레이어
  * 병원 A: 2개 토큰 / 병원 B: 3개 토큰 / 병원 C: 3개 토큰 생성
* **Top Model (Global Transformer & Classifier):**
  * 각 병원의 토큰 8개 + `[CLS]` 토큰 1개 = **총 9개 토큰 시퀀스** 입력
  * **Linear Attention ($\phi(x) = x^2$ 또는 PolyActivation)** 기반 Transformer Block
  * Final Binary Classification Head

### 3.2 Linear Attention 채택 이유
$$\text{Attention}(Q, K, V) = \phi(Q) \left( \phi(K)^T V \right), \quad \text{where } \phi(x) = x^2$$

1. **Softmax 제거:** 지수함수($e^x$) 및 나눗셈을 배제하여 다항식 근사 오버헤드 제거
2. **연산 복잡도 단축:** $O(N^2 \cdot d) \rightarrow O(N \cdot d^2)$로 감소 ($N=9$)
3. **CKKS 연산 깊이(Depth) 절감:** 곱셈 Depth 2~3 내 해결 가능 $\rightarrow$ **Bootstrapping 불필요**
4. **SecretFlow SPU 통신 최적화:** SPU 엔진의 비버 트리플(Beaver's Triple) 기반 행렬 곱셈만으로 순전파/역전파 완료

---

## 4. 학습 파이프라인 (Training Phase: SecretFlow SPU MPC)

학습 단계에서는 동형암호의 속도 한계를 극복하고 수직 분할된 데이터를 안전하게 공동 학습하기 위해 **SecretFlow SPU(Secure Processing Unit) 기반 MPC**를 적용합니다. 

* **VFL Data Loading:** SecretFlow의 `VDataLoader`를 이용하여 3개 병원의 수직 분할 데이터를 병렬 로드하고 SecretSharing 기반 상태로 변환
* **SPU Compiler & Protocols:** PyTorch/JAX 기반 FT-Transformer 연산 그래프를 SPU 엔진(ABY3, Cheetah, Semi2k 등 프로토콜 지원)으로 컴파일하여 병원 간 비밀 분산 연산을 실행
* **안정성 및 최신성:** 지속적으로 업데이트되는 SecretFlow 생태계를 활용하여 GPU 가속 및 최신 PyTorch 파이프라인과의 호환성 유지