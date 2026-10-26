"""병원별 프로세스로 실행하는 SPU 학습 (프로덕션 모드, 병원당 포트 2개).

loss는 bce_clip_z_dp 하나만 쓴다:
  grad_z = (sigmoid_approx(clip(z, -4, 4)) - y) / batch  +  DP 노이즈
  노이즈는 세 병원이 각자 N(0, dp_sigma^2/3)을 뽑아 SPU에 넣고, SPU 안에서 그래디언트에
  더한 뒤에야 공개한다 (top_layer_jax._loss_and_noisy_grad_bce_clip_z).

train_variant_spu.py와 같은 14개 피처 / 역할 분배를 쓰지만,
한 프로세스가 전체 데이터를 읽고 메모리에서 나누는 대신:

  - 각 병원은 자기 PYU 안에서 자기 파일(mpc/hospital_data/{A,B,C}.csv)만 읽는다.
    표준화(StandardScaler)도 열 단위라 각자 로컬에서 한다.
  - 라벨은 병원 B(hospital1)만 갖는다. y_pyu.to(spu)가 곧 비밀분산이므로
    additive_split을 손으로 할 필요가 없다.
  - 가중치와 Adam 상태는 각 병원 PYU 안에만 있다. grad_i는 SPU에서 병원 i에게만
    공개되고(grads_spu[i].to(pyus[i])), 업데이트도 병원 i 안에서 끝난다 --
    train_variant_spu.py처럼 sf.reveal로 드라이버에 모으지 않는다.
  - 테스트 정확도도 SPU 안에서 계산해 정확도 스칼라만 공개한다.

공개되는 값: 행 개수 n(PSI 이후 공개 메타데이터로 가정), 배치 인덱스(공개 시드),
배치 loss(--reveal-loss일 때만), 테스트 정확도.

전제: 세 파일이 같은 nsrrid 순서로 정렬돼 있다 (prepare_hospital_data.py의 출력).
실제로는 PSI로 맞춰야 한다.

WSL 전용 (venv_wsl). 병원 3개를 각각 다른 터미널에서 실행하거나 run_spu_parties.sh 사용:
  python -m mpc.train_spu_party --party hospital0 --epochs 30 --max-batches 1
  python -m mpc.train_spu_party --party hospital1 --epochs 30 --max-batches 1
  python -m mpc.train_spu_party --party hospital2 --epochs 30 --max-batches 1
세 프로세스의 인자(에폭, 배치, 시드, dp-sigma 등)는 반드시 같아야 한다.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import jax.numpy as jnp
import secretflow as sf
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import SHHS_LABEL, AHI_THRESHOLD
from top_layer_jax import _loss_and_noisy_grad_bce_clip_z
from mpc.spu_cluster import DEFAULT_CONFIG, load_cluster_config, init_party

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hospital_data")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOSS_TYPE = "bce_clip_z_dp"  # 체크포인트 메타데이터용

# 학습 포인트
# 노이즈 생성 -> spu 계산 -> 노이즈 섞인 gradient를 병원 i 에게만 공개 -> Adam 업데이트

# ── 각 병원 PYU 안에서만 실행되는 함수들 ──────────────────────────────────────
# PYU - Python Processing Unit (각 병원에서 발생하는 연산)

def _load_features(role, data_dir):
    """(원본 X, 피처 이름). 표준화는 학습/test 분할 뒤 _standardize에서 학습 행만으로 맞춘다."""
    df = pd.read_csv(os.path.join(data_dir, f"{role}.csv"), index_col="nsrrid")
    df = df.drop(columns=[SHHS_LABEL], errors="ignore")
    return df.values.astype(np.float32), list(df.columns)


def _standardize(X_raw, names, idx_train):
    """(표준화된 X, 표준화 메타데이터). 평균·표준편차는 학습 행으로만 계산해 test 정보가 섞이지 않게 한다.
    메타데이터는 체크포인트에 같이 저장해 추론 때 같은 변환을 쓴다."""
    scaler = StandardScaler().fit(X_raw[idx_train])
    X = scaler.transform(X_raw).astype(np.float32)
    meta = {"feature_names": names, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    return X, meta


def _load_label(role, data_dir):
    df = pd.read_csv(os.path.join(data_dir, f"{role}.csv"), index_col="nsrrid")
    return (df[SHHS_LABEL].values >= AHI_THRESHOLD).astype(np.float32)


def _num_rows(X):
    return int(X.shape[0])


def _init_state(X, has_bias, seed):
    # nn.Linear(n_i, 1, bias=False)의 기본 초기화와 같은 분포: U(-1/sqrt(n_i), 1/sqrt(n_i))
    rng = np.random.default_rng(seed)
    n_feat = X.shape[1]
    bound = 1.0 / np.sqrt(n_feat)
    params = {"w": rng.uniform(-bound, bound, size=(n_feat, 1)).astype(np.float32)}
    if has_bias:
        params["b"] = np.zeros((1,), dtype=np.float32) # 해당 bias는 병원 b만 가짐 
    return {
        "params": params,
        "m": {k: np.zeros_like(v) for k, v in params.items()}, # Adam Optimizer가 지금까지 gradient가 어떤 방향으로 어느정도 움직였는지
        "v": {k: np.zeros_like(v) for k, v in params.items()}, # 기억하는 공간임
        "t": 0,
    }


def _logit_share(state, X, idx): # 로짓 조각을 계산하는 함수 
    out = X[idx] @ state["params"]["w"] # 한 배치당 256명의 사람들이 있음
    if "b" in state["params"]: # 병원 b 에서만 bias를 더함 
        out = out + state["params"]["b"]
    return jnp.array(out.astype(np.float32))# SPU로 보낼 수 있는 JAX 배열로 바꿈


def _batch_label(y, idx):
    return jnp.array(y[idx][:, None])


def _noise_share(shape, sigma):
    return jnp.array(np.random.normal(0.0, sigma, size=shape).astype(np.float32))

# spu에서 받은 그래디언트로 각 병원이 자기 가중치를 한 번 업데이트하는 함수이다. 
def _adam_step(state, X, idx, grad_logit, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """torch.optim.Adam 기본값과 같은 업데이트. grad_logit = dL/d(logit_share_i), (B, 1)."""
    g = np.asarray(grad_logit, dtype=np.float32) # grad_logit은 환자마다 로짓을 얼마나 바꿔야 하는지 이다. 
    grads = {"w": X[idx].T @ g} # 그  피처값 (5x256)@(256x1)  그 환자의 로짓 그래디언트 
    if "b" in state["params"]:
        grads["b"] = g.sum(axis=0)

    t = state["t"] + 1
    params, m, v = {}, {}, {}
    for k, p in state["params"].items():
        m[k] = beta1 * state["m"][k] + (1 - beta1) * grads[k] # gradient 이동 평균(방향 부드럽게 함) 
        v[k] = beta2 * state["v"][k] + (1 - beta2) * grads[k] ** 2 # gradient 제곱의 이동 평균(크기 정규화함)
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        params[k] = (p - lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32) #학습률
    return {"params": params, "m": m, "v": v, "t": t}


def _save_checkpoint(state, meta, path, info):
    """병원 i 자신의 가중치·표준화 값만 자기 파일에 쓴다 (다른 병원 것은 이 프로세스에 없다)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = state["params"]
    arrays = {
        "w": params["w"][:, 0],
        "feature_names": np.array(meta["feature_names"]),
        "scaler_mean": np.asarray(meta["scaler_mean"], dtype=np.float64),
        "scaler_scale": np.asarray(meta["scaler_scale"], dtype=np.float64),
        **{k: np.array(v) for k, v in info.items()},
    }
    if "b" in params:
        arrays["bias"] = params["b"]
    # 임시 파일에 다 쓴 뒤 교체한다 -- 저장 도중 프로세스가 죽어도 이전 체크포인트는 온전히 남는다.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return path


# ── SPU 안에서 실행되는 함수들 ────────────────────────────────────────────────
# 라벨은 병원 B 한 곳에서 오므로 y_shares 자리에 [y] 하나만 넘긴다 (sum([y]) == y).
# spu 내에서 하는 것 : 로짓 조각 합침 -> 범위 자르기(clip_z) -> 시그모이드 근사 -> 로짓 그래디언트 -> dp 노이즈 더하기
def _spu_step(logit_shares, y, noise_shares):
    """(loss, grad_0 + noise, ..., grad_{n-1} + noise). 노이즈는 공개 전에 SPU 안에서 더해진다."""
    return _loss_and_noisy_grad_bce_clip_z(logit_shares, [y], noise_shares)
# 해당 함수는 SPU 진입점이다. logit_share [z_A, z_B, z_C] 각각 (256x1)
# y (병원B에만 있는 label 값), noise_share  [n_A,n_B,n_C] (256x1) 세 병원의 noise_share 

def _spu_accuracy(logit_shares, y):
    logit = sum(logit_shares)
    return jnp.mean((logit > 0) == (y > 0.5))


# ── 학습 루프 (세 병원 프로세스가 모두 같은 코드를 실행한다) ─────────────────

def run(self_party, config, n_epochs, batch_size, max_batches_per_epoch, dp_sigma, lr, eval_every,
        seed=0, reveal_loss=False, data_dir=DATA_DIR, ckpt_dir=CKPT_DIR, n_folds=None, fold=None):
    party_names = list(config["parties"])
    n_parties = len(party_names)
    roles = [config["parties"][p]["role"] for p in party_names]
    label_idx = party_names.index(config["label_holder"])
    bias_idx = party_names.index(config["bias_holder"])

    print(f"[{self_party}] SecretFlow 프로덕션 모드 초기화 (ABY3, {n_parties}자)", flush=True)
    for p in party_names:
        c = config["parties"][p]
        print(f"  {p} (role {c['role']}): sf={c['host']}:{c['sf_port']}  spu={c['host']}:{c['spu_port']}")
    spu, pyus = init_party(self_party, config)
    print(f"[{self_party}] loss_type={LOSS_TYPE}, dp_sigma={dp_sigma}"
          + ("" if reveal_loss else "  (loss 비공개)"), flush=True)

    # 1) 각자 자기 파일만 로드
    loaded = [pyus[i](_load_features, num_returns=2)(roles[i], data_dir) for i in range(n_parties)]
    X_raw = [x for x, _ in loaded]
    names = [m for _, m in loaded]
    y = pyus[label_idx](_load_label)(roles[label_idx], data_dir)

    # 행 개수는 공개 메타데이터 -- 세 병원이 같은 환자 집합을 같은 순서로 갖는지 확인
    n_rows = sf.reveal([pyus[i](_num_rows)(X_raw[i]) for i in range(n_parties)])
    assert len(set(n_rows)) == 1, f"병원별 행 개수가 다르다: {n_rows} (PSI 정렬 필요)"
    n = n_rows[0]
    if n_folds:
        # 교차검증: 공개 시드로 행 번호만 나눈다. 라벨을 쓰는 stratified 분할은 병원 B 외에는 할 수 없다.
        splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(np.arange(n)))
        idx_train, idx_test = splits[fold]
        print(f"[data] n={n}, fold {fold+1}/{n_folds}, train={len(idx_train)}, test={len(idx_test)}", flush=True)
    else:
        idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=42)
        print(f"[data] n={n}, train={len(idx_train)}, test={len(idx_test)}", flush=True)

    # 표준화: 각 병원이 자기 학습 행만으로 평균·표준편차를 맞춘다 (test 행은 변환만)
    scaled = [pyus[i](_standardize, num_returns=2)(X_raw[i], names[i], idx_train) for i in range(n_parties)]
    X = [x for x, _ in scaled]
    meta = [m for _, m in scaled]

    # 2) 병원별 로컬 파라미터 초기화 (bias는 한 병원만)
    states = [pyus[i](_init_state)(X[i], i == bias_idx, seed * 100 + i) for i in range(n_parties)]

    per_party_sigma = dp_sigma / np.sqrt(n_parties) if dp_sigma > 0 else 0.0
    shuffle_rng = np.random.default_rng(seed)  # 공개 시드 -- 세 프로세스가 같은 배치 순서를 쓴다

    def evaluate():
        test_logits = [pyus[i](_logit_share)(states[i], X[i], idx_test).to(spu) for i in range(n_parties)]
        y_test = pyus[label_idx](_batch_label)(y, idx_test).to(spu)
        return float(sf.reveal(spu(_spu_accuracy)(test_logits, y_test)))

    def save_checkpoints(epoch, test_acc):
        """평가할 때마다 덮어쓴다. 학습을 중간에 멈춰도 마지막 평가 시점의 가중치가 남는다.
        세 프로세스가 같은 에폭에서 같이 호출하고, 각 병원은 자기 파일만 쓴다."""
        saved = []
        for i in range(n_parties):
            info = {"party": party_names[i], "role": roles[i], "loss_type": LOSS_TYPE, "dp_sigma": dp_sigma,
                    "epochs": epoch, "epochs_planned": n_epochs, "seed": seed, "test_acc": test_acc}
            if n_folds:
                info.update({"n_folds": n_folds, "fold": fold})
            path = os.path.join(ckpt_dir, f"{party_names[i]}.npz")
            saved.append(pyus[i](_save_checkpoint)(states[i], meta[i], path, info))
        sf.wait(saved)
        print(f"[{self_party}] 체크포인트 저장 (epoch {epoch}): "
              f"{os.path.join(ckpt_dir, self_party + '.npz')}", flush=True)

    # 학습 시작 부분 
    t_train_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        perm = idx_train[shuffle_rng.permutation(len(idx_train))]
        epoch_loss, n_batches = 0.0, 0
        t_epoch_start = time.perf_counter()

        for start in range(0, len(perm), batch_size):
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                break
            idx = perm[start:start + batch_size]
            if len(idx) < batch_size:
                break

            logit_spu = [pyus[i](_logit_share)(states[i], X[i], idx).to(spu) for i in range(n_parties)]
            y_spu = pyus[label_idx](_batch_label)(y, idx).to(spu)

            # 각 병원이 N(0, dp_sigma^2/3)을 로컬에서 뽑는다. 합은 SPU 안에서만 만들어진다.
            noise_spu = [pyus[i](_noise_share)((batch_size, 1), per_party_sigma).to(spu)
                         for i in range(n_parties)]
            outs = spu(
                _spu_step,
                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                user_specified_num_returns=1 + n_parties,
            )(logit_spu, y_spu, noise_spu)
            loss_spu, grads_spu = outs[0], list(outs[1:]) # 결과 loss 와 gradient를 얻음

            for i in range(n_parties):
                # dp 노이즈가 섞인 grad_i가 병원 i에게만 공개된다.
                grad_i = grads_spu[i].to(pyus[i])
                states[i] = pyus[i](_adam_step)(states[i], X[i], idx, grad_i, lr) 
                # 자신의 가중치를 업데이트함

            # reveal_loss=False면 loss는 끝까지 비밀분산 상태로 남는다 (reveal 자체를 안 함).
            if reveal_loss:
                epoch_loss += float(sf.reveal(loss_spu))
            n_batches += 1

        epoch_wall_s = time.perf_counter() - t_epoch_start
        loss_str = f"{epoch_loss/max(n_batches,1):.4f}" if reveal_loss else "비공개"
        msg = f"  Epoch {epoch:3d}/{n_epochs} | loss={loss_str} | wall={epoch_wall_s:.1f}s ({n_batches} batches)"
        
        if epoch % eval_every == 0 or epoch == 1 or epoch == n_epochs:
            test_acc = evaluate()
            msg += f" | test_acc={test_acc:.4f}"
        print(msg, flush=True)
        if ckpt_dir and (epoch % eval_every == 0 or epoch == n_epochs):
            save_checkpoints(epoch, test_acc)

    print(f"[{self_party}] 학습 완료 — 총 {time.perf_counter()-t_train_start:.1f}s", flush=True)
    sf.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--party", required=True, help="이 프로세스가 맡을 병원 (예: hospital0)")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--ckpt-dir", default=CKPT_DIR, help="병원별 체크포인트 저장 폴더 (빈 문자열이면 저장 안 함). "
                        "--eval-every 에폭마다 덮어쓰므로 중간에 멈춰도 마지막 평가 시점 가중치가 남는다")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--dp-sigma", type=float, default=0.01)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reveal-loss", action="store_true",
                   help="디버깅용으로 배치 loss를 공개한다 (기본은 비공개)")
    p.add_argument("--n-folds", type=int, default=None, help="교차검증 fold 수 (없으면 80/20 분할)")
    p.add_argument("--fold", type=int, default=0, help="--n-folds를 줄 때 test로 쓸 fold 번호 (0부터)")
    a = p.parse_args()
    run(a.party, load_cluster_config(a.config), a.epochs, a.batch_size, a.max_batches, a.dp_sigma,
        a.lr, a.eval_every, a.seed, a.reveal_loss, a.data_dir, a.ckpt_dir, a.n_folds, a.fold)
