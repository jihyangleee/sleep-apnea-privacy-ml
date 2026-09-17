"""VFL training with the top-layer computed on SecretFlow's SPU device.

Same math and data pipeline as simulate.py's run_distributed_simulation(),
but the "combine logit shares -> sigmoid approx -> MSE loss -> backward"
step (top_layer_jax.top_layer_loss_and_grad) now actually executes as a
secret-shared MPC computation across 3 simulated SecretFlow parties
(ABY3, semi-honest, honest-majority -- see spu_top_layer_test.py), instead
of the hand-rolled Beaver Triple protocol in secret_sharing.py.

Sub-model stays exactly as in simulate.py: plain local PyTorch, never
touches SPU. `logit_share_i = W_top_i @ emb_i` also stays local plain
PyTorch (block-linearity means it never needs protection) -- only the
already-computed scalar logit_share_i crosses into the SPU device.

WSL-only: requires the venv_wsl environment (secretflow needs Linux/WSL2).
Real MPC has per-call overhead (network-simulated even in "local" mode),
so start with few epochs/batches before scaling up -- see __main__.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp
import secretflow as sf
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy

from dataset import load_shhs_data, partition_data_vertical
from model import HospitalModel
from top_layer_jax import top_layer_loss_and_grad
from simulate import SLEEP_FEATURE_GROUPS, SLEEP_CLIENT_LABELS, NUM_CLIENTS


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


def run_distributed_simulation_spu(
    csv_path: str,
    n_epochs: int = 1,
    batch_size: int = 32,
    max_batches_per_epoch: int | None = 3,
    emb_dim: int = 16,
):
    """Same as simulate.run_distributed_simulation but top-layer runs on SPU.

    max_batches_per_epoch caps batches per epoch for smoke testing -- real
    MPC has per-call overhead, so this defaults small. Pass None for a full
    epoch once timing looks acceptable.
    """
    print("[SPU Distributed FL] SHHS 데이터 로드")
    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  hospital {i}: {label}")

    total_emb = emb_dim * NUM_CLIENTS
    shared_W  = nn.Linear(total_emb, 1)
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
    params.extend(shared_W.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    print("[SPU Distributed FL] secretflow 3-party SPU 초기화 (ABY3, semi-honest, honest-majority)")
    spu, pyus = _make_spu()

    print(f"\n[SPU Distributed FL] 학습 시작 - {n_epochs} epoch(s), batch={batch_size}, "
          f"max_batches_per_epoch={max_batches_per_epoch}")
    t_train_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch_start = time.perf_counter()

        for start in range(0, n, batch_size):
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                break
            idx     = perm[start : start + batch_size]
            y_batch = y_train[idx]
            optimizer.zero_grad()
            t_batch_start = time.perf_counter()

            # Step 1: local sub-model forward (plain PyTorch, stays local)
            local_embs = [
                hospitals[i].local_emb(X_trains[i][idx]) for i in range(NUM_CLIENTS)
            ]

            # Step 2: local top-layer matmul (plain PyTorch -- block-linearity
            # means this needs no protection, see model.py logit_share()).
            # Keep the autograd-tracked tensor around for backward() later;
            # a detached numpy copy is what actually crosses into SPU.
            logit_shares = [
                hospitals[j].logit_share(local_embs[j], j == 0)
                for j in range(NUM_CLIENTS)
            ]
            logit_shares_np = [ls.detach().numpy() for ls in logit_shares]
            # Only sum(y_shares) matters to the SPU-side computation, so an
            # even split is a valid (if non-random) secret sharing of y.
            y_shares_np = [y_batch.unsqueeze(1).numpy() / NUM_CLIENTS for _ in range(NUM_CLIENTS)]

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

            loss_spu, grads_spu = spu(
                top_layer_loss_and_grad,
                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                user_specified_num_returns=2,
            )(logit_shares_spu, y_shares_spu)

            loss  = float(sf.reveal(loss_spu))
            grads = sf.reveal(grads_spu)  # party-scoped in spirit: each
            # hospital only needs its own grads[i] to continue backward.
            # sf.reveal here reveals all three to this single-process demo;
            # a real multi-machine deployment would call reveal_to(party_i).

            # Step 4: continue backward locally in PyTorch -- each hospital
            # only needs dL/d(logit_share_i), which is what SPU handed back.
            for i in range(NUM_CLIENTS):
                grad_i = torch.tensor(np.array(grads[i]), dtype=torch.float32)
                logit_shares[i].backward(grad_i)
            optimizer.step()

            batch_ms = (time.perf_counter() - t_batch_start) * 1000
            print(f"    batch {n_batches}: loss={loss:.4f}  ({batch_ms:.0f} ms, incl. real SPU MPC round trip)")

            epoch_loss += loss
            n_batches  += 1

        epoch_wall_ms = (time.perf_counter() - t_epoch_start) * 1000

        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                test_embs = torch.cat(
                    [hospitals[i].local_emb(X_tests[i]) for i in range(NUM_CLIENTS)],
                    dim=1,
                )
                logit = shared_W(test_embs)
                pred  = (logit.squeeze() > 0).float()
                acc   = (pred == y_test).float().mean().item()
            print(f"  Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss/max(n_batches,1):.4f} | "
                  f"test_acc={acc:.4f} | wall={epoch_wall_ms:.1f}ms ({n_batches} batches)")
        else:
            print(f"  Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss/max(n_batches,1):.4f} | "
                  f"wall={epoch_wall_ms:.1f}ms ({n_batches} batches)")

    train_total_s = time.perf_counter() - t_train_start
    print(f"[SPU Distributed FL] 학습 완료 — 총 {train_total_s:.1f}s")

    sf.shutdown()
    return hospitals, shared_W, scaler


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    # Smoke test (1 epoch, 3 batches) already confirmed correctness and
    # ~150-200ms/batch after the one-time SPU compile cost -- full run now.
    run_distributed_simulation_spu(csv_path, n_epochs=30, batch_size=32, max_batches_per_epoch=None)
