import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from dataset import generate_galaxy_watch_users
from simulate import run_distributed_simulation, SLEEP_FEATURE_GROUPS
from he_client import build_he_context
from model import HospitalModel

MODEL_PATH = "vertical_model.pt"

ENTRY_LABELS = ["Hospital A", "Hospital B", "Hospital C"]


# ── Distributed FL (SS + DP noise, linear top-model) ──────────────────────────

def run_distributed_fl(
    csv_path: str = None,
    dreamt_dir: str = None,
    dp_sigma: float = 0.01,
):
    """Fully distributed VFL training — single linear top-model, no Beaver Triple."""
    hospitals, shared_W, scaler = run_distributed_simulation(
        csv_path, dreamt_dir, n_epochs=30, dp_sigma=dp_sigma
    )

    emb_dim = hospitals[0].emb_dim

    torch.save(
        {
            "mode":           "distributed",
            "feature_groups": SLEEP_FEATURE_GROUPS,
            "emb_dim":        emb_dim,
            "sub_0":          hospitals[0].sub.state_dict(),
            "sub_1":          hospitals[1].sub.state_dict(),
            "sub_2":          hospitals[2].sub.state_dict(),
            "top_W":          shared_W.state_dict(),
            "scaler_mean":    scaler.mean_.tolist(),
            "scaler_scale":   scaler.scale_.tolist(),
        },
        MODEL_PATH,
    )
    print(f"[Distributed FL] model saved -> {MODEL_PATH}")


# ── HE Inference ──────────────────────────────────────────────────────────────

def _load_hospitals_for_he(ckpt: dict):
    """Reconstruct HospitalModel list and set per-hospital HE weight slices."""
    feature_groups = ckpt["feature_groups"]
    emb_dim        = ckpt["emb_dim"]
    total_emb      = emb_dim * len(feature_groups)

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

    print("[HE Inference] 워치 → 병원별 partial feature 암호화, 단일 linear top-model\n")

    for idx, (x, name) in enumerate(zip(X_gw, profile_names)):
        # Plaintext reference
        with torch.no_grad():
            local_embs  = [
                hospitals[i].sub(
                    torch.tensor(x[hospitals[i].feature_groups[i]], dtype=torch.float32).unsqueeze(0)
                )
                for i in range(len(hospitals))
            ]
            cat_emb     = torch.cat(local_embs, dim=1)
            plain_logit = hospitals[0].top_W(cat_emb).item()
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
    parser.add_argument("--csv",    default=None, help="SHHS-1 CSV 경로")
    parser.add_argument("--dreamt", default=None, help="DREAMT v2.1.0 루트 경로")
    parser.add_argument(
        "--dp-sigma",
        type=float,
        default=0.01,
        help="DP noise sigma (default 0.01; 0 = 비활성화)",
    )
    args = parser.parse_args()

    if args.mode == "distributed":
        run_distributed_fl(args.csv, args.dreamt, dp_sigma=args.dp_sigma)
    elif args.mode == "he-infer":
        run_he_infer()
