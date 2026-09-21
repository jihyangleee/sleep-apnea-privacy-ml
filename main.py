import argparse
import random
import sys
import numpy as np
import torch
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from dataset import generate_galaxy_watch_users
from simulate import run_distributed_simulation, save_checkpoint
from he_client import build_he_context
from model import LinearHospitalModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # cp949 콘솔에서 한글/특수문자 출력 오류 방지

MODEL_PATH = "vertical_model.pt"

ENTRY_LABELS = ["Hospital A", "Hospital B", "Hospital C"]


# ── Distributed FL (SS + DP noise, linear top-model) ──────────────────────────
# 학습 과정
# 로지스틱 회귀 
# 병원별 logit share를 합산함
# 병원 0 이 스칼라 bias를 하나 더한다.
# 시그모이드를 적용한다. 
def run_distributed_fl(
    csv_path: str,
    dp_sigma: float = 0.01,
    n_epochs: int = 30,
):
    """Fully distributed VFL training — vertical logistic regression, no Beaver Triple for the linear part."""
    hospitals, shared_W, scaler = run_distributed_simulation(
        csv_path, n_epochs=n_epochs, dp_sigma=dp_sigma, model_type="linear"
    )

    save_checkpoint(MODEL_PATH, hospitals, shared_W, scaler, "linear")
    print(f"[Distributed FL] model saved -> {MODEL_PATH}")


# ── HE Inference ──────────────────────────────────────────────────────────────# 
# 학습 결과 생성된 .pt 파일을 torch.load를 통해 dict로 나타내고
# 이를 사용할 때는 ckpt라는 변수를 이용한다. 
def _load_hospitals_for_he(ckpt: dict):
    """Reconstruct the hospital list and set per-hospital HE weight slices."""
    feature_groups = ckpt["feature_groups"]
    model_type     = ckpt.get("model_type")
    if model_type != "linear":
        raise ValueError(
            f"지원하지 않는 체크포인트 model_type={model_type!r}: "
            "linear(수직 로지스틱 회귀) 체크포인트만 로드할 수 있습니다."
        )
    # 병원 객체 3개를 만듦 
    hospitals = [LinearHospitalModel(i, feature_groups) for i in range(len(feature_groups))]
    # 각 병원은 self.sub = nn.Linear(5,1, bias=False)를 가짐 
    # 해당 시점의 가중치는 랜덤 초기값이다. 
    for i, h in enumerate(hospitals):
        h.sub.load_state_dict(ckpt[f"sub_{i}"])  
        h.eval()
    b_top = ckpt["bias"].detach().numpy()           # (1,)
    # 학습된 bias를 numpy로 꺼내 각 병원의 HE용 가중치를 미리 계산함 
    for h in hospitals:
        h.build_he_weights(b_top) # build_he_weights가 모든 병원에 bias를 넣는다.
        # 학습 시에는 병원0만 bias를 갖지만 추론 시에는 진입 병원이 랜덤이라 
        # 어느 병원이 와도 bias를 한 번 더할 수 있게 한 구성이다.  
    return hospitals


def _plain_logit(hospitals, ckpt: dict, x: np.ndarray) -> float:
    """Plaintext reference logit for one standardized sample."""
    with torch.no_grad():
        shares = [
            h.local_emb(torch.tensor(x[h.feature_groups[h.id]], dtype=torch.float32).unsqueeze(0))
            for h in hospitals
        ]
        # 각 각 병원의 특성 5개만 뽑는다.  => .unsqueeze(0)을 통해 텐서로 바꾸고 배치 차원을 붙여
        # (1,5)로 만듦
        return float(sum(shares).item() + ckpt["bias"].item())


def run_he_infer(model_path: str = MODEL_PATH):
    """Distributed HE inference demo.

    Watch (patient device):
      - holds CKKS secret key
      - encrypts each hospital's feature slice separately (VFL privacy)
      - decrypts returned enc(logit) → probability

    Each hospital:
      - receives only its own enc(features_i)
      - enc(features_i) · w_i → enc(logit_i)  (plaintext matmul, no activation)
      - coordinator sums all enc(logit_i) + b

    Privacy: patient data never decrypted at any hospital.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    mode = ckpt.get("mode", "distributed")
    print(f"\n[HE Inference] checkpoint mode: {mode}")

    hospitals = _load_hospitals_for_he(ckpt)

    scaler                = StandardScaler()
    scaler.mean_          = np.array(ckpt["scaler_mean"])
    scaler.scale_         = np.array(ckpt["scaler_scale"])
    scaler.n_features_in_ = len(scaler.mean_)

    # 학습 때 쓴 표준화 통계를 체크 포인트에서 꺼내 복원함
    # (추론 입력도 학습 때와 마찬가지로 같은 기준으로 표준화해야하기 때문)
    # 데모용 워치 사용자 5명의 원시 특성을 만들고 표준화함 
    raw_X, profile_names = generate_galaxy_watch_users()
    X_gw = scaler.transform(raw_X).astype(np.float32)

    # CKKS 컨텍스트를 두 개 만든다.
    # 이때, 환자용은 비밀키가 있고, 병원용은 비밀키만 뺀 것이다. - 병원은 복호화를 못함  
    patient_ctx  = build_he_context()
    hospital_ctx = ts.context_from(patient_ctx.serialize(save_secret_key=False))

    print(f"[HE Inference] 워치 → 병원별 partial feature 암호화 (vertical logistic regression)\n")

    for idx, (x, name) in enumerate(zip(X_gw, profile_names)):
        # Plaintext reference
        plain_logit = _plain_logit(hospitals, ckpt, x)
        plain_prob  = float(1.0 / (1.0 + np.exp(-plain_logit)))

        # Watch encrypts each hospital's feature slice (zero-padded to power-of-2)
        all_enc_xi = {}
        for h in hospitals:
            raw = x[h.feature_groups[h.id]].astype(np.float64)
            pad_to = max(1 << (len(raw) - 1).bit_length(), 4)
            padded = np.zeros(pad_to)
            padded[:len(raw)] = raw
            all_enc_xi[h.id] = ts.ckks_vector(patient_ctx, padded.tolist()).serialize()

        # Randomly select entry-point hospital
        # 진입 병원을 랜덤 선택
        entry_id = random.randrange(len(hospitals))
        others   = [h for h in hospitals if h.id != entry_id]
        # 추론 단계에서는 각 병우너이 enc(logit_h)를 계산한 후 
        # 진입 병원이 세 개를 더하고 bias를 더한다. enc(z)를 직렬화해 반환 
        enc_result = hospitals[entry_id].run_he_inference(all_enc_xi, hospital_ctx, others)
        # 워치가 복호화해서 비교 
        he_prob    = LinearHospitalModel.decrypt_result(enc_result, patient_ctx)
        # patient_ct로 복화화 가능 -> sigmoid 씌워서 확률을 얻는다. 
        diff       = abs(plain_prob - he_prob)

        print(f"  Profile {idx}: {name}")
        print(f"    Plaintext prob  : {plain_prob:.4f}")
        print(f"    HE prob         : {he_prob:.4f}  |err|={diff:.6f}  (entry={ENTRY_LABELS[entry_id]})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["distributed", "he-infer"],
        required=True,
        help=(
            "distributed : SS + DP noise, 단일 linear top-model\n"
            "he-infer    : HE 암호화 추론 데모"
        ),
    )
    parser.add_argument("--csv", default=None, help="SHHS-1 CSV 경로 (--mode distributed 시 필수)")
    parser.add_argument(
        "--dp-sigma",
        type=float,
        default=0.01,
        help="DP noise sigma (default 0.01; 0 = 비활성화)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="학습 epoch 수 (default 30)")
    parser.add_argument("--ckpt", default=MODEL_PATH, help="he-infer 에서 읽을 체크포인트 (default vertical_model.pt)")
    args = parser.parse_args()

    if args.mode == "distributed":
        if args.csv is None:
            parser.error("--mode distributed 에는 --csv 가 필수입니다.")
        run_distributed_fl(args.csv, dp_sigma=args.dp_sigma, n_epochs=args.epochs)
    elif args.mode == "he-infer":
        run_he_infer(args.ckpt)
