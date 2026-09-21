"""VFL training with the top-layer computed on SecretFlow's SPU device.

Same math and data pipeline as simulate.py's run_distributed_simulation(),
but the "combine logit shares -> sigmoid approx -> MSE loss -> backward"
step (top_layer_jax.top_layer_loss_and_grad) executes as a secret-shared MPC
computation across 3 simulated SecretFlow parties (ABY3, semi-honest,
honest-majority -- see spu_top_layer_test.py), instead of the hand-rolled
Beaver Triple protocol in secret_sharing.py.

What runs where
---------------
  local, plain PyTorch : sub-model / linear weights -> logit_share_i
                         (block-linearity: needs no protection)
  SPU (secret-shared)  : sum(logit_share_i) -> sigmoid approx -> pred - y
                         -> MSE -> d(loss)/d(logit_share_i)
  revealed             : the scalar loss (monitoring) and, to hospital i only,
                         its own d(loss)/d(logit_share_i) (+ Gaussian DP noise,
                         same as simulate.py). Nothing else leaves the SPU.

Labels are additively secret-shared with random shares (additive_split) before
entering the SPU, so no party ever holds y (or a scaled copy of it).

model_type "linear" (default) is the vertical logistic regression
(LinearHospitalModel); "mlp" is the sub-model MLP + shared linear top.

WSL-only: requires the venv_wsl environment (secretflow needs Linux/WSL2).
Real MPC has per-call overhead, and the first batch pays a one-time ray/SPU
start-up + compile cost (minutes on /mnt/d). Every batch must have the same
shape (the last partial batch is dropped) so the SPU program compiles once.

    python -m mpc.simulate_spu --epochs 100 --out vertical_model.pt
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp
import secretflow as sf
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy

from dataset import load_shhs_data, partition_data_vertical
from model import HospitalModel, LinearHospitalModel
from secret_sharing import additive_split
from top_layer_jax import top_layer_loss_and_grad
from simulate import SLEEP_FEATURE_GROUPS, SLEEP_CLIENT_LABELS, NUM_CLIENTS, save_checkpoint


def _make_spu():
    try:
        sf.shutdown()
    except Exception:
        pass
    parties = [f"hospital{i}" for i in range(NUM_CLIENTS)]
    sf.init(parties=parties, address="local")
    cluster_def = sf.utils.testing.cluster_def(parties=parties)
    # sf.SPU()'s json.dumps(cluster_def['runtime_config']) chokes on the raw
    # spu.ProtocolKind/FieldType enums this testing helper returns in this
    # secretflow/spu version pairing -- swap for their .name strings.
    cluster_def["runtime_config"] = {
        k: (v.name if hasattr(v, "name") else v)
        for k, v in cluster_def["runtime_config"].items()
    }
    spu  = sf.SPU(cluster_def)
    pyus = [sf.PYU(p) for p in parties]
    return spu, pyus


def _loss_and_per_party_grads(logit_shares, y_shares):
    """SPU program: returns (loss, grad_0, ..., grad_{n-1}) as separate outputs so
    each grad_i can be revealed to hospital i alone (a single stacked grads
    object could only be revealed to everyone at once)."""
    loss, grads = top_layer_loss_and_grad(logit_shares, y_shares)
    return (loss, *grads)


def run_distributed_simulation_spu(
    csv_path: str,
    n_epochs: int = 100,
    batch_size: int = 256,
    max_batches_per_epoch: int | None = None,
    emb_dim: int = 16,
    model_type: str = "linear",
    dp_sigma: float = 0.01,
    lr: float = 1e-2,
    eval_every: int = 10,
    checkpoint_path: str | None = None,
):
    """Same as simulate.run_distributed_simulation but the top-layer runs on SPU.

    Larger batches than simulate.py's 32 by default: each SPU call has fixed
    overhead, so fewer, bigger batches are much cheaper (lr is raised to match).
    max_batches_per_epoch caps batches per epoch for smoke testing.
    checkpoint_path: if set, save a checkpoint (same format as main.py).
    """
    assert model_type in ("linear", "mlp"), model_type

    print("[SPU Distributed FL] SHHS 데이터 로드")
    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  hospital {i}: {label}")

    if model_type == "linear":
        shared_W  = None
        hospitals = [LinearHospitalModel(i, SLEEP_FEATURE_GROUPS) for i in range(NUM_CLIENTS)]
    else:
        shared_W  = nn.Linear(emb_dim * NUM_CLIENTS, 1)
        hospitals = [
            HospitalModel(i, SLEEP_FEATURE_GROUPS, emb_dim, shared_W)
            for i in range(NUM_CLIENTS)
        ]

    X_trains = [
        torch.tensor(partitions[i]["X_train"], dtype=torch.float32)
        for i in range(NUM_CLIENTS)
    ]
    X_tests = [
        torch.tensor(partitions[i]["X_test"], dtype=torch.float32)
        for i in range(NUM_CLIENTS)
    ]
    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    y_test  = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)

    params = []
    for h in hospitals:
        params.extend(h.sub.parameters())
    if shared_W is not None:
        params.extend(shared_W.parameters())
    else:
        params.append(hospitals[0].bias)
    optimizer = torch.optim.Adam(params, lr=lr)

    print(f"[SPU Distributed FL] model_type={model_type} | batch={batch_size} lr={lr} "
          f"dp_sigma={dp_sigma} | train n={n}")
    print("[SPU Distributed FL] secretflow 3-party SPU 초기화 (ABY3, semi-honest, honest-majority)")
    spu, pyus = _make_spu()

    print(f"\n[SPU Distributed FL] 학습 시작 - {n_epochs} epoch(s)")
    t_train_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch_start = time.perf_counter()

        for start in range(0, n, batch_size):
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                break
            idx = perm[start : start + batch_size]
            if len(idx) < batch_size:
                break   # drop the partial last batch: constant shape -> SPU compiles once
            y_batch = y_train[idx]
            optimizer.zero_grad()

            # Step 1: local forward (plain PyTorch, stays local). For linear,
            # local_emb is already this hospital's logit share w_i . x_i.
            local_embs = [
                hospitals[i].local_emb(X_trains[i][idx]) for i in range(NUM_CLIENTS)
            ]

            # Step 2: local logit share (block-linearity -> no protection needed).
            # Keep the autograd-tracked tensor for backward(); only a detached
            # numpy copy crosses into the SPU.
            logit_shares = [
                hospitals[j].logit_share(local_embs[j], j == 0)
                for j in range(NUM_CLIENTS)
            ]
            logit_shares_np = [ls.detach().numpy() for ls in logit_shares]
            # Labels: random additive shares (sum = y), so no party holds y.
            y_shares_np = [s.numpy() for s in additive_split(y_batch.unsqueeze(1), n=NUM_CLIENTS)]

            # Step 3: hand off to SPU -- THIS is the actual secret-shared MPC
            # boundary. Each hospital's PYU places its own private value;
            # .to(spu) is where it becomes a secret-shared SPU object.
            logit_shares_pyu = [
                pyus[i](lambda x: jnp.array(x))(logit_shares_np[i])
                for i in range(NUM_CLIENTS)
            ]
            y_shares_pyu = [
                pyus[i](lambda x: jnp.array(x))(y_shares_np[i])
                for i in range(NUM_CLIENTS)
            ]
            logit_shares_spu = [x.to(spu) for x in logit_shares_pyu]
            y_shares_spu     = [x.to(spu) for x in y_shares_pyu]

            outs = spu(
                _loss_and_per_party_grads,
                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                user_specified_num_returns=1 + NUM_CLIENTS,
            )(logit_shares_spu, y_shares_spu)
            loss_spu, grads_spu = outs[0], list(outs[1:])

            loss = float(sf.reveal(loss_spu))
            # Party-scoped reveal: hospital i receives only its own gradient.
            # (sf.reveal below just fetches each party's value to this single-
            # process driver; in a multi-machine deployment each party would
            # read its own PYU object locally.)
            grads = sf.reveal([grads_spu[i].to(pyus[i]) for i in range(NUM_CLIENTS)])

            # Step 4: continue backward locally in PyTorch. DP noise on the
            # returned gradient blunts label inference from (pred - y).
            for i in range(NUM_CLIENTS):
                grad_i = torch.tensor(np.array(grads[i]), dtype=torch.float32)
                if dp_sigma > 0:
                    grad_i = grad_i + torch.randn_like(grad_i) * dp_sigma
                logit_shares[i].backward(grad_i)
            optimizer.step()

            epoch_loss += loss
            n_batches  += 1

        epoch_wall_s = time.perf_counter() - t_epoch_start
        msg = (f"  Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss/max(n_batches,1):.4f} | "
               f"wall={epoch_wall_s:.1f}s ({n_batches} batches)")

        if epoch % eval_every == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                if shared_W is None:
                    logit = sum(hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)) + hospitals[0].bias
                else:
                    test_embs = torch.cat(
                        [hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)],
                        dim=1,
                    )
                    logit = shared_W(test_embs)
                pred = (logit.squeeze() > 0).float()
                acc  = (pred == y_test).float().mean().item()
            msg += f" | test_acc={acc:.4f}"
        print(msg, flush=True)

    train_total_s = time.perf_counter() - t_train_start
    print(f"[SPU Distributed FL] 학습 완료 — 총 {train_total_s:.1f}s")

    if checkpoint_path:
        save_checkpoint(checkpoint_path, hospitals, shared_W, scaler, model_type)
        print(f"[SPU Distributed FL] model saved -> {checkpoint_path} (model_type={model_type})")

    sf.shutdown()
    return hospitals, shared_W, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="shhs/datasets/shhs1-dataset-0.21.0.csv")
    parser.add_argument("--model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dp-sigma", type=float, default=0.01, help="0 = 비활성화")
    parser.add_argument("--max-batches", type=int, default=None, help="epoch당 최대 배치 수 (스모크 테스트용)")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--out", default=None, help="체크포인트 저장 경로 (예: vertical_model.pt)")
    a = parser.parse_args()
    run_distributed_simulation_spu(
        a.csv, n_epochs=a.epochs, batch_size=a.batch_size,
        max_batches_per_epoch=a.max_batches, model_type=a.model,
        dp_sigma=a.dp_sigma, lr=a.lr, eval_every=a.eval_every, checkpoint_path=a.out,
    )
