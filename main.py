import argparse
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from dataset import generate_galaxy_watch_users
from simulate import run_distributed_simulation, SLEEP_FEATURE_GROUPS
from he_client import build_he_context
from model import HospitalModel, LinearHospitalModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # cp949 콘솔에서 한글/특수문자 출력 오류 방지

MODEL_PATH = "vertical_model.pt"

ENTRY_LABELS = ["Hospital A", "Hospital B", "Hospital C"]


# ── Distributed FL (SS + DP noise, linear top-model) ──────────────────────────

def run_distributed_fl(
    csv_path: str,
    dp_sigma: float = 0.01,
    model_type: str = "linear",
    n_epochs: int = 30,
):
    """Fully distributed VFL training — no Beaver Triple for the linear/sub-model part.

    model_type "linear": vertical logistic regression (default).
    model_type "mlp"   : sub-model MLP + shared linear top.
    """
    hospitals, shared_W, scaler = run_distributed_simulation(
        csv_path, n_epochs=n_epochs, dp_sigma=dp_sigma, model_type=model_type
    )

    ckpt = {
        "mode":           "distributed",
        "model_type":     model_type,
        "feature_groups": SLEEP_FEATURE_GROUPS,
        "emb_dim":        hospitals[0].emb_dim,
        "scaler_mean":    scaler.mean_.tolist(),
        "scaler_scale":   scaler.scale_.tolist(),
    }
    for i, h in enumerate(hospitals):
        ckpt[f"sub_{i}"] = h.sub.state_dict()
    if model_type == "linear":
        ckpt["bias"] = hospitals[0].bias.detach().clone()
    else:
        ckpt["top_W"] = shared_W.state_dict()

    torch.save(ckpt, MODEL_PATH)
    print(f"[Distributed FL] model saved -> {MODEL_PATH} (model_type={model_type})")


# ── HE Inference ──────────────────────────────────────────────────────────────

def _load_hospitals_for_he(ckpt: dict):
    """Reconstruct the hospital list and set per-hospital HE weight slices."""
    feature_groups = ckpt["feature_groups"]
    model_type     = ckpt.get("model_type", "mlp")   # old checkpoints predate model_type

    if model_type == "linear":
        hospitals = [LinearHospitalModel(i, feature_groups) for i in range(len(feature_groups))]
        for i, h in enumerate(hospitals):
            h.sub.load_state_dict(ckpt[f"sub_{i}"])
            h.eval()
        b_top = ckpt["bias"].detach().numpy()          # (1,)
        for h in hospitals:
            h.build_he_weights(b_top)
        return hospitals

    emb_dim   = ckpt["emb_dim"]
    total_emb = emb_dim * len(feature_groups)

    shared_W = nn.Linear(total_emb, 1)
    shared_W.load_state_dict(ckpt["top_W"])

    hospitals = [
        HospitalModel(i, feature_groups, emb_dim, shared_W)
        for i in range(len(feature_groups))
    ]
    for i, h in enumerate(hospitals):
        h.sub.load_state_dict(ckpt[f"sub_{i}"])

    for h in hospitals:
        h.eval()

    # Each hospital holds its own column slice; bias is added once when summed
    W_top = shared_W.weight.detach().numpy()   # (1, total_emb)
    b_top = shared_W.bias.detach().numpy()     # (1,)
    for i, h in enumerate(hospitals):
        W_top_i = W_top[:, i * emb_dim : (i + 1) * emb_dim]  # (1, emb_dim)
        h.build_he_weights(W_top_i, b_top)

    return hospitals


def _plain_logit(hospitals, ckpt: dict, x: np.ndarray) -> float:
    """Plaintext reference logit for one standardized sample."""
    with torch.no_grad():
        embs = [
            h.local_emb(torch.tensor(x[h.feature_groups[h.id]], dtype=torch.float32).unsqueeze(0))
            for h in hospitals
        ]
        if ckpt.get("model_type", "mlp") == "linear":
            return float(sum(embs).item() + ckpt["bias"].item())
        return hospitals[0].top_W(torch.cat(embs, dim=1)).item()


def run_he_infer():
    """Distributed HE inference demo.

    Watch (patient device):
      - holds CKKS secret key
      - encrypts each hospital's feature slice separately (VFL privacy)
      - decrypts returned enc(logit) → probability

    Each hospital:
      - receives only its own enc(features_i)
      - sub_model: enc(features_i) → enc(emb_i)  (-2 HE levels, cubic act)
      - applies W_top_i column slice → enc(logit_i)  (no extra level)
      - coordinator sums all enc(logit_i) + b_top

    Privacy: patient data never decrypted at any hospital.
    """
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    mode = ckpt.get("mode", "distributed")
    print(f"\n[HE Inference] checkpoint mode: {mode}")

    hospitals = _load_hospitals_for_he(ckpt)

    scaler                = StandardScaler()
    scaler.mean_          = np.array(ckpt["scaler_mean"])
    scaler.scale_         = np.array(ckpt["scaler_scale"])
    scaler.n_features_in_ = len(scaler.mean_)

    raw_X, profile_names = generate_galaxy_watch_users()
    X_gw = scaler.transform(raw_X).astype(np.float32)

    patient_ctx  = build_he_context()
    hospital_ctx = ts.context_from(patient_ctx.serialize(save_secret_key=False))

    print(f"[HE Inference] 워치 → 병원별 partial feature 암호화 (model_type={ckpt.get('model_type', 'mlp')})\n")

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
        entry_id = random.randrange(len(hospitals))
        others   = [h for h in hospitals if h.id != entry_id]

        enc_result = hospitals[entry_id].run_he_inference(all_enc_xi, hospital_ctx, others)
        he_prob    = HospitalModel.decrypt_result(enc_result, patient_ctx)
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
    parser.add_argument(
        "--model",
        choices=["linear", "mlp"],
        default="linear",
        help="linear: 수직 로지스틱 회귀 (default) | mlp: sub-model MLP + linear top",
    )
    parser.add_argument("--epochs", type=int, default=30, help="학습 epoch 수 (default 30)")
    args = parser.parse_args()

    if args.mode == "distributed":
        if args.csv is None:
            parser.error("--mode distributed 에는 --csv 가 필수입니다.")
        run_distributed_fl(args.csv, dp_sigma=args.dp_sigma, model_type=args.model, n_epochs=args.epochs)
    elif args.mode == "he-infer":
        run_he_infer()
