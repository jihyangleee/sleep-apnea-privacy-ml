import argparse
import numpy as np
import torch
import tenseal as ts

from dataset import load_shhs_data
from simulate import run_vertical_simulation
from he_client import HEInference, build_he_context
from model import VerticalHeartNet

MODEL_PATH = "vertical_model.pt"


def run_vertical_fl(csv_path: str):
    """Vertical FL 학습 후 모델 저장."""
    model = run_vertical_simulation(csv_path, n_epochs=30)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_groups":   model.feature_groups,
            "emb_dim":          model.emb_dim,
        },
        MODEL_PATH,
    )
    print(f"[Vertical FL] 모델 저장 완료 → {MODEL_PATH}")


def run_he_infer(csv_path: str):
    """학습된 평문 중앙 모델에 신규 개인이 HE 암호화된 데이터로 추론 요청하는 데모.

    역할 구분:
      - 신규 개인(환자): feature 전체(9개)를 CKKS로 암호화 → 서버에 전송
                        복호화된 결과로 예측 확률 확인 (비밀키 보유)
      - 서버           : 평문 가중치 보유, 공개키로만 HE 연산 → 암호화된 로짓 반환
                        개인의 feature 평문을 볼 수 없음
    """
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = VerticalHeartNet(ckpt["feature_groups"], ckpt["emb_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    he_module = HEInference(model)

    X, y = load_shhs_data(csv_path)
    X_sample = X[:5]
    y_sample = y[:5]

    # 개인: 비밀키 포함 컨텍스트 생성
    individual_ctx = build_he_context()
    # 서버: 공개키만 전달 (비밀키 제거)
    server_ctx = ts.context_from(individual_ctx.serialize(save_secret_key=False))

    print("\n[HE Inference] 신규 개인 → 암호화된 추론 데모")
    print(f"{'샘플':<6} {'실제 레이블':<12} {'평문 추론':<12} {'HE 추론':<12} {'일치'}")
    print("-" * 55)

    for i, (x, label) in enumerate(zip(X_sample, y_sample)):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            plain_prob = torch.sigmoid(model(x_t)).item()

        enc_bytes  = HEInference.encrypt_input(x, individual_ctx)
        enc_result = he_module.run_he_inference(enc_bytes, server_ctx)
        he_prob    = HEInference.decrypt_result(enc_result, individual_ctx)

        match = "✓" if abs(plain_prob - he_prob) < 0.01 else "✗"
        print(f"  {i:<4} {int(label):<12} {plain_prob:.4f}      {he_prob:.4f}      {match}")

    print("\n[HE Inference] 완료 — 서버는 개인 feature를 평문으로 보지 않았습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["vertical", "he-infer"],
        required=True,
        help="vertical: Vertical FL 학습 후 모델 저장 | he-infer: HE 암호화 입력으로 프라이빗 추론 데모",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="NSRR SHHS-1 CSV 파일 경로 (예: shhs1-dataset-0.21.0.csv)",
    )
    args = parser.parse_args()

    if args.mode == "vertical":
        run_vertical_fl(args.csv)
    elif args.mode == "he-infer":
        run_he_infer(args.csv)
