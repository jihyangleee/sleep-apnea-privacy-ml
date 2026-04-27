import numpy as np
import tenseal as ts

from model import VerticalHeartNet


def build_he_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    return ctx


# ── HE Inference ─────────────────────────────────────────────────────────────
#
# Vertical FL로 학습된 평문 중앙 모델에
# 학습에 참여하지 않은 신규 개인(환자)이 자신의 데이터를 CKKS로 암호화해 전송하고
# 서버가 암호화된 상태로 연산하여 암호화된 결과를 반환하는 프라이빗 추론 모듈.
#
# 보호 대상: 신규 개인의 입력 feature (서버가 평문을 볼 수 없음)
# 평문 보유: 서버의 모델 가중치
#
# HE 연산 깊이:
#   서브모델 Square (1레벨) + 탑모델 Square (1레벨) = 총 2레벨 소비
#   coeff_mod_bit_sizes=[60,40,40,60] → 3레벨 제공 → 충분

class HEInference:
    """평문 VerticalHeartNet + CKKS 암호화 입력으로 프라이빗 추론.

    신규 개인이 자신의 feature 전체(13개)를 CKKS로 암호화해 전달하면,
    서버는 평문 가중치와 HE 연산만으로 암호화된 로짓을 반환한다.
    개인은 자신의 비밀키로 복호화해 예측 확률을 확인한다.
    """

    def __init__(self, model: VerticalHeartNet):
        self.model = model.eval()
        self.feature_groups = model.feature_groups
        self.emb_dim = model.emb_dim
        self.n_features = sum(len(fg) for fg in self.feature_groups)
        self.total_emb = self.emb_dim * len(self.feature_groups)
        self._build_extended_weights()

    def _build_extended_weights(self):
        """각 서브모델 가중치를 n_features × total_emb 크기로 확장(zero-padding).

        enc_x (n_features,).mm_(W_ext) 한 번으로 모든 서브모델의 선형 변환을 처리.
        i번 서브모델의 출력은 결과 벡터의 [i*emb_dim : (i+1)*emb_dim] 구간에만 배치된다.
        """
        self.W_ext_list = []
        self.b_ext_list = []

        for i, (sub, fg) in enumerate(zip(self.model.sub_models, self.feature_groups)):
            W = sub.linear.weight.detach().numpy()  # (emb_dim, |fg|)
            b = sub.linear.bias.detach().numpy()    # (emb_dim,)

            W_ext = np.zeros((self.n_features, self.total_emb), dtype=np.float64)
            b_ext = np.zeros(self.total_emb, dtype=np.float64)

            start = i * self.emb_dim
            for out_idx in range(self.emb_dim):
                for j, fi in enumerate(fg):
                    W_ext[fi, start + out_idx] = float(W[out_idx, j])
            b_ext[start:start + self.emb_dim] = b.astype(np.float64)

            self.W_ext_list.append(W_ext.tolist())
            self.b_ext_list.append(b_ext.tolist())

        # 탑모델: mm_은 enc_v @ matrix이므로 weight.T를 전달
        W1 = self.model.top_model.linear1.weight.detach().numpy()   # (32, total_emb)
        self.W_top1_T = W1.T.astype(np.float64).tolist()            # (total_emb, 32)
        self.b_top1 = self.model.top_model.linear1.bias.detach().numpy().astype(np.float64).tolist()

        W2 = self.model.top_model.linear2.weight.detach().numpy()   # (1, 32)
        self.W_top2_T = W2.T.astype(np.float64).tolist()            # (32, 1)
        self.b_top2 = self.model.top_model.linear2.bias.detach().numpy().astype(np.float64).tolist()

    # ── 개인(환자) 측 실행 ───────────────────────────────────────────────────

    @staticmethod
    def encrypt_input(x: np.ndarray, context: ts.Context) -> bytes:
        """신규 개인의 feature 벡터(13개) 전체를 CKKS로 암호화."""
        return ts.ckks_vector(context, x.astype(np.float64).tolist()).serialize()

    @staticmethod
    def decrypt_result(enc_bytes: bytes, context: ts.Context) -> float:
        """서버 반환 암호화 로짓을 복호화 후 sigmoid 적용 → 예측 확률."""
        vec = ts.lazy_ckks_vector_from(enc_bytes)
        vec.link_context(context)
        logit = vec.decrypt()[0]
        return float(1.0 / (1.0 + np.exp(-logit)))

    # ── 서버 측 실행 (공개키만 사용) ────────────────────────────────────────

    def run_he_inference(self, enc_input_bytes: bytes, context: ts.Context) -> bytes:
        """암호화된 입력으로 추론 수행 — 서버는 개인의 평문 feature를 볼 수 없다.

        1. 각 서브모델: enc_x @ W_ext_i + b_ext_i → Square  (enc_emb_i)
           - W_ext_i는 zero-padding되어 i번 서브모델 담당 구간에만 값이 있음
        2. enc_concat = enc_emb_0 + enc_emb_1 + enc_emb_2
           (구간이 겹치지 않으므로 element-wise 합 = concatenation과 동일)
        3. 탑모델: Linear → Square → Linear → enc_logit
        """
        enc_x_bytes = enc_input_bytes  # mm_이 in-place이므로 서브모델마다 새 복사본 사용

        enc_emb_parts = []
        for W_ext, b_ext in zip(self.W_ext_list, self.b_ext_list):
            enc_xi = ts.lazy_ckks_vector_from(enc_x_bytes)
            enc_xi.link_context(context)
            enc_xi.mm_(W_ext)                        # enc_x @ W_ext (ciphertext × plaintext)
            enc_xi += b_ext
            enc_emb_parts.append(enc_xi * enc_xi)    # Square activation (레벨 1 소비)

        # 임베딩 연결 (zero-padding 구간이 겹치지 않음)
        enc_concat = enc_emb_parts[0]
        for part in enc_emb_parts[1:]:
            enc_concat = enc_concat + part

        # 탑모델
        enc_concat.mm_(self.W_top1_T)                # (total_emb,) → (32,)
        enc_concat += self.b_top1
        enc_h_sq = enc_concat * enc_concat            # Square activation (레벨 1 소비)
        enc_h_sq.mm_(self.W_top2_T)                   # (32,) → (1,)
        enc_h_sq += self.b_top2

        return enc_h_sq.serialize()
