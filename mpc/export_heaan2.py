"""SPU 학습 결과(mpc/checkpoints/hospital{0,1,2}.npz)를 HEaaN2 추론용 텍스트 모델로 변환한다.

출력 형식 (HEAAN2_LINEAR_V1):
  HEAAN2_LINEAR_V1
  NUM_FEATURES 14
  FEATURE_NAMES <MODEL_FEATURES 순서>
  SCALER_MEAN  <14개, FEATURE_NAMES 순서>
  SCALER_SCALE <14개, FEATURE_NAMES 순서>
  HOSPITAL_i_INDICES <병원 i 피처의 FEATURE_NAMES 내 인덱스>
  HOSPITAL_i_WEIGHTS <같은 순서의 가중치>
  BIAS <bias>

각 npz에는 자기 피처의 이름·표준화 값·가중치만 들어 있으므로, 피처 이름으로
MODEL_FEATURES 상의 위치를 찾아 하나의 파일로 합친다. 세 병원 가중치를 한 파일로
모으는 단계이므로 학습 후 모델 배포 시점에 한 번만 실행한다.

사용법: python -m mpc.export_heaan2 [--ckpt-dir mpc/checkpoints] [--out mpc/checkpoints/heaan2_model.txt]
"""

import argparse
import os

import numpy as np

from dataset import MODEL_FEATURES

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
N_HOSPITALS = 3


def _fmt(values):
    return " ".join(repr(float(v)) for v in values)


def export(ckpt_dir, out_path):
    ckpts = [np.load(os.path.join(ckpt_dir, f"hospital{i}.npz")) for i in range(N_HOSPITALS)]

    # 세 체크포인트가 같은 학습에서 나왔는지 확인 (다른 실행의 가중치를 섞으면 안 된다)
    for key in ("loss_type", "epochs", "seed"):
        vals = {str(c[key]) for c in ckpts if key in c.files}
        assert len(vals) <= 1, f"병원별 체크포인트의 {key}가 다르다: {vals}"

    scaler_mean = np.full(len(MODEL_FEATURES), np.nan)
    scaler_scale = np.full(len(MODEL_FEATURES), np.nan)
    hospital_lines, bias = [], None
    for i, c in enumerate(ckpts):
        names = [str(n) for n in c["feature_names"]]
        idx = [MODEL_FEATURES.index(n) for n in names]
        scaler_mean[idx] = c["scaler_mean"]
        scaler_scale[idx] = c["scaler_scale"]
        hospital_lines.append(f"HOSPITAL_{i}_INDICES " + " ".join(map(str, idx)))
        hospital_lines.append(f"HOSPITAL_{i}_WEIGHTS " + _fmt(c["w"]))
        if "bias" in c.files:
            assert bias is None, "bias를 가진 병원이 둘 이상이다"
            bias = float(c["bias"][0])

    missing = [MODEL_FEATURES[j] for j in np.flatnonzero(np.isnan(scaler_mean))]
    assert not missing, f"어느 병원에도 없는 피처: {missing}"
    assert bias is not None, "bias를 가진 병원이 없다"

    lines = [
        "HEAAN2_LINEAR_V1",
        f"NUM_FEATURES {len(MODEL_FEATURES)}",
        "FEATURE_NAMES " + " ".join(MODEL_FEATURES),
        "SCALER_MEAN " + _fmt(scaler_mean),
        "SCALER_SCALE " + _fmt(scaler_scale),
        *hospital_lines,
        f"BIAS {bias!r}",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    info = ckpts[0]
    print(f"[export] {out_path}")
    print(f"  loss_type={info['loss_type']}  epochs={info['epochs']}  test_acc={float(info['test_acc']):.4f}")


def export_v5(model_txt, out_dir, data_dir, test_size=0.2, seed=42):
    """heaan2_model.txt → inference_v5(DistributedInferenceV5.cpp)가 읽는 파일들.

      client_model.txt        HEAAN2_CLIENT_V1: 피처 이름, 스케일러, bias (환자 기기용)
      hospital{i}_model.txt   HEAAN2_HOSPITAL_V1: 병원 i의 피처 인덱스와 가중치
      A.csv / B.csv / C.csv   학습과 같은 test 분할(train_spu_party.py와 같은 random_state)의 원본 피처
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    tok = {}
    with open(model_txt, encoding="utf-8") as fh:
        for line in fh.read().splitlines()[1:]:
            if line.strip():
                k, *v = line.split()
                tok[k] = v
    os.makedirs(out_dir, exist_ok=True)

    def write(name, lines):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")

    write("client_model.txt", [
        "HEAAN2_CLIENT_V1",
        "NUM_FEATURES " + tok["NUM_FEATURES"][0],
        "FEATURE_NAMES " + " ".join(tok["FEATURE_NAMES"]),
        "SCALER_MEAN " + " ".join(tok["SCALER_MEAN"]),
        "SCALER_SCALE " + " ".join(tok["SCALER_SCALE"]),
        "BIAS " + tok["BIAS"][0],
    ])
    for i in range(N_HOSPITALS):
        write(f"hospital{i}_model.txt", [
            "HEAAN2_HOSPITAL_V1",
            f"HOSPITAL_ID {i}",
            "INDICES " + " ".join(tok[f"HOSPITAL_{i}_INDICES"]),
            "WEIGHTS " + " ".join(tok[f"HOSPITAL_{i}_WEIGHTS"]),
        ])

    # 학습 test 분할 재현: 세 병원 파일은 같은 nsrrid 순서이고, train_spu_party.py가 행 번호로 나눈다.
    roles = ["A", "B", "C"]
    frames = [pd.read_csv(os.path.join(data_dir, f"{r}.csv")) for r in roles]
    _, idx_test = train_test_split(np.arange(len(frames[0])), test_size=test_size, random_state=seed)
    for r, df in zip(roles, frames):
        df.iloc[np.sort(idx_test)].to_csv(os.path.join(out_dir, f"{r}.csv"), index=False)
    print(f"[export_v5] {out_dir}: client_model.txt, hospital0~2_model.txt, A/B/C.csv (test {len(idx_test)}명)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--out", default=os.path.join(CKPT_DIR, "heaan2_model.txt"))
    p.add_argument("--v5-dir", default="", help="지정하면 inference_v5 형식 모델 파일과 test CSV도 쓴다")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(CKPT_DIR), "hospital_data"))
    a = p.parse_args()
    export(a.ckpt_dir, a.out)
    if a.v5_dir:
        export_v5(a.out, a.v5_dir, a.data_dir)
