import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from dataset import generate_galaxy_watch_users
from simulate import run_vertical_simulation, run_distributed_simulation, SLEEP_FEATURE_GROUPS
from he_client import build_he_context
from model import VerticalHeartNet, HospitalModel

MODEL_PATH = "vertical_model.pt"

ENTRY_LABELS = ["Hospital A", "Hospital B", "Hospital C"]


# ── Old vertical FL (SS + DH masking, semi-honest server) ────────────────────

def run_vertical_fl(csv_path: str = None, dreamt_dir: str = None):
    """Vertical FL 학습 후 모델 + 스케일러 저장 (semi-honest server 가정)."""
    model, scaler = run_vertical_simulation(csv_path, dreamt_dir, n_epochs=30)
    torch.save(
        {
            "mode":             "vertical",
            "model_state_dict": model.state_dict(),
            "feature_groups":   model.feature_groups,
            "emb_dim":          model.emb_dim,
            "scaler_mean":      scaler.mean_.tolist(),
            "scaler_scale":     scaler.scale_.tolist(),
        },
        MODEL_PATH,
    )
    print(f"[Vertical FL] model saved -> {MODEL_PATH}")


# ── New distributed FL (SS + DH masking + DP noise + Beaver Triple) ──────────

def run_distributed_fl(
    csv_path: str = None,
    dreamt_dir: str = None,
    dp_sigma: float = 0.01,
):
    """Fully distributed VFL training — no central server, Beaver Triple MPC."""
    hospitals, shared_W1, shared_W2, scaler = run_distributed_simulation(
        csv_path, dreamt_dir, n_epochs=30, dp_sigma=dp_sigma
    )

    top_hidden = shared_W1.out_features
    emb_dim    = hospitals[0].emb_dim

    torch.save(
        {
            "mode":          "distributed",
            "feature_groups": SLEEP_FEATURE_GROUPS,
            "emb_dim":        emb_dim,
            "top_hidden":     top_hidden,
            "sub_0":          hospitals[0].sub.state_dict(),
            "sub_1":          hospitals[1].sub.state_dict(),
            "sub_2":          hospitals[2].sub.state_dict(),
            "top_W1":         shared_W1.state_dict(),
            "top_W2":         shared_W2.state_dict(),
            "scaler_mean":    scaler.mean_.tolist(),
            "scaler_scale":   scaler.scale_.tolist(),
        },
        MODEL_PATH,
    )
    print(f"[Distributed FL] model saved -> {MODEL_PATH}")


# ── HE Inference — works with both checkpoint formats ────────────────────────

def _split_top_weights_for_he(hospitals):
    """Additively split W1, b1, W2, b2 across hospitals for SS inference.

    No hospital ever holds the full top-model weights during inference.
    sum(W1_share_i) = W1,  sum(b1_share_i) = b1,  etc.
    """
    import numpy as np

    n  = len(hospitals)
    W1 = hospitals[0].top_W1.weight.detach().numpy()   # (top_hidden, total_emb)
    b1 = hospitals[0].top_W1.bias.detach().numpy()     # (top_hidden,)
    W2 = hospitals[0].top_W2.weight.detach().numpy()   # (1, top_hidden)
    b2 = hospitals[0].top_W2.bias.detach().numpy()     # (1,)

    def np_additive_split(arr):
        noise = [np.random.randn(*arr.shape).astype(np.float64) for _ in range(n - 1)]
        return noise + [arr.astype(np.float64) - sum(noise)]

    W1_T_shares = np_additive_split(W1.T)   # each: (total_emb, top_hidden)
    b1_shares   = np_additive_split(b1)     # each: (top_hidden,)
    W2_T_shares = np_additive_split(W2.T)   # each: (top_hidden, 1)
    b2_shares   = np_additive_split(b2)     # each: (1,)

    for i, h in enumerate(hospitals):
        h.build_he_weights(
            W1_T_share=W1_T_shares[i],
            b1_share=b1_shares[i],
            W2_T_share=W2_T_shares[i],
            b2_share=b2_shares[i],
        )


def _load_hospitals_for_he(ckpt: dict):
    """Reconstruct HospitalModel list and split top-model weights for SS inference."""
    feature_groups = ckpt["feature_groups"]
    emb_dim        = ckpt["emb_dim"]

    if ckpt.get("mode") == "distributed":
        top_hidden = ckpt["top_hidden"]
        total_emb  = emb_dim * len(feature_groups)
        shared_W1  = nn.Linear(total_emb, top_hidden)
        shared_W2  = nn.Linear(top_hidden, 1)
        shared_W1.load_state_dict(ckpt["top_W1"])
        shared_W2.load_state_dict(ckpt["top_W2"])

        hospitals = [
            HospitalModel(i, feature_groups, emb_dim, top_hidden, shared_W1, shared_W2)
            for i in range(len(feature_groups))
        ]
        for i, h in enumerate(hospitals):
            h.sub.load_state_dict(ckpt[f"sub_{i}"])
    else:
        # Legacy vertical checkpoint: wrap VerticalHeartNet into HospitalModel
        legacy = VerticalHeartNet(feature_groups, emb_dim)
        legacy.load_state_dict(ckpt["model_state_dict"])

        total_emb  = emb_dim * len(feature_groups)
        top_hidden = legacy.top_model.linear1.out_features
        shared_W1  = nn.Linear(total_emb, top_hidden)
        shared_W2  = nn.Linear(top_hidden, 1)
        shared_W1.load_state_dict(legacy.top_model.linear1.state_dict())
        shared_W2.load_state_dict(legacy.top_model.linear2.state_dict())

        hospitals = [
            HospitalModel(i, feature_groups, emb_dim, top_hidden, shared_W1, shared_W2)
            for i in range(len(feature_groups))
        ]
        for i, h in enumerate(hospitals):
            h.sub.load_state_dict(legacy.sub_models[i].state_dict())

    for h in hospitals:
        h.eval()

    # Split top-model weights into additive shares — no hospital holds full W1/W2
    _split_top_weights_for_he(hospitals)
    return hospitals


def run_he_infer():
    """Distributed HE inference demo.

    Watch (patient device):
      - holds CKKS secret key
      - splits features by hospital assignment, encrypts each slice separately
      - sends enc(features_i) to hospital i — each hospital sees only its own slice
      - decrypts returned enc(logit) -> probability

    Each hospital:
      - receives only its own enc(features_i), computes enc(emb_i) via sub-model
      - exchanges enc(emb_i) with other hospitals (ciphertext, unreadable)
      - applies its additive W1_share to all enc(emb_j) -> enc(h_share_i)
      - coordinator sums enc(h_share_i) -> enc(h_linear)
      - coordinator applies PolyAct in CKKS (ciphertext, nothing revealed)
      - each hospital applies W2_share -> enc(logit_share_i); coordinator sums

    Privacy:
      - Sub-model: each hospital sees only its own encrypted feature slice
      - Top-model: W1 and W2 are additively split; no hospital holds the full weights
      - Patient data never decrypted at any hospital
    """
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    mode = ckpt.get("mode", "vertical")
    print(f"\n[HE Inference] checkpoint mode: {mode}")

    hospitals = _load_hospitals_for_he(ckpt)

    scaler               = StandardScaler()
    scaler.mean_         = np.array(ckpt["scaler_mean"])
    scaler.scale_        = np.array(ckpt["scaler_scale"])
    scaler.n_features_in_ = len(scaler.mean_)

    raw_X, profile_names = generate_galaxy_watch_users()
    X_gw = scaler.transform(raw_X).astype(np.float32)

    # Patient context (with secret key); hospital context (public key only)
    patient_ctx  = build_he_context()
    hospital_ctx = ts.context_from(patient_ctx.serialize(save_secret_key=False))

    print("[HE Inference] 워치 -> 병원별 partial feature 암호화 전송, W1/W2 additive share\n")

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
            h1          = hospitals[0].top_W1(cat_emb)
            h1_act      = h1 * (h1 + 0.5)
            plain_logit = hospitals[0].top_W2(h1_act).item()
            plain_prob  = float(1.0 / (1.0 + np.exp(-plain_logit)))

        # Watch encrypts each hospital's feature slice separately
        all_enc_xi = {
            h.id: HospitalModel.encrypt_feature_slice(
                x[h.feature_groups[h.id]], patient_ctx
            )
            for h in hospitals
        }

        # Randomly select entry-point hospital for this request
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
        choices=["vertical", "distributed", "he-infer"],
        required=True,
        help=(
            "vertical    : 기존 Vertical FL (semi-honest server)\n"
            "distributed : Beaver Triple MPC + DP noise (서버 없음)\n"
            "he-infer    : HE 암호화 추론 데모 (두 모드 모두 호환)"
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

    if args.mode == "vertical":
        run_vertical_fl(args.csv, args.dreamt)
    elif args.mode == "distributed":
        run_distributed_fl(args.csv, args.dreamt, dp_sigma=args.dp_sigma)
    elif args.mode == "he-infer":
        run_he_infer()
