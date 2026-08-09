"""Training loop for the linear-attention FT-Transformer architecture
(linear_attention.md). Plain JAX for the top-model here (no SPU yet) --
validates the new architecture end-to-end on real SHHS data before paying
the cost of wiring it onto SPU's secret-shared device, same order of
operations used for the previous top_layer_jax.py -> spu_top_layer_test.py
-> simulate_spu.py progression.

Sub-model: each hospital's FeatureTokenizer (model.py) — local, plaintext
PyTorch, one token per raw feature. Never touches MPC/HE (block-local, same
reasoning as the old ClientSubModel).

Top-model: linear_attention_jax.py's linear-attention transformer over the
9-token sequence (8 feature tokens + [CLS]). Plain JAX for now; wiring onto
sf.spu is the next step once this is confirmed to train correctly.

d(loss)/d(feature_tokens) is what would get party-scoped-revealed per
hospital once this runs on SPU -- here it's simply sliced locally since
everything's in one process.
"""

import sys
import time
import numpy as np
import torch
import jax.numpy as jnp

# Windows console defaults to cp949 when stdout is redirected to a file,
# which can't encode the em-dashes in these print statements -- force UTF-8
# regardless of how this script is invoked.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import load_shhs_data, partition_data_vertical
from model import FeatureTokenizer
from simulate import SLEEP_FEATURE_GROUPS, SLEEP_CLIENT_LABELS, NUM_CLIENTS
from linear_attention_jax import init_params, linear_attention_loss_and_grad, linear_attention_forward, sigmoid_approx
import jax


def run_linear_attention_simulation(
    csv_path: str,
    n_epochs: int = 30,
    batch_size: int = 32,
    d: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
):
    print("[Linear-Attention FT-Transformer] SHHS 데이터 로드")
    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )
    for i, label in enumerate(SLEEP_CLIENT_LABELS):
        print(f"  hospital {i}: {label}  ({len(SLEEP_FEATURE_GROUPS[i])} tokens)")

    tokenizers = [
        FeatureTokenizer(len(SLEEP_FEATURE_GROUPS[i]), d) for i in range(NUM_CLIENTS)
    ]
    tok_optimizer = torch.optim.Adam(
        [p for tok in tokenizers for p in tok.parameters()], lr=lr
    )

    key = jax.random.PRNGKey(seed)
    top_params = init_params(key, d)

    X_trains = [
        torch.tensor(partitions[i]["X_train"], dtype=torch.float32)
        for i in range(NUM_CLIENTS)
    ]
    X_tests = [
        torch.tensor(partitions[i]["X_test"], dtype=torch.float32)
        for i in range(NUM_CLIENTS)
    ]
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)

    # Manual Adam state for the JAX-side top-model params (no optax dependency).
    m_state = {k: jnp.zeros_like(v) for k, v in top_params.items()}
    v_state = {k: jnp.zeros_like(v) for k, v in top_params.items()}
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    step = 0

    def adam_update(params, grads, m, v, step):
        step += 1
        new_params, new_m, new_v = {}, {}, {}
        for k in params:
            g = grads[k]
            new_m[k] = beta1 * m[k] + (1 - beta1) * g
            new_v[k] = beta2 * v[k] + (1 - beta2) * (g ** 2)
            m_hat = new_m[k] / (1 - beta1 ** step)
            v_hat = new_v[k] / (1 - beta2 ** step)
            new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)
        return new_params, new_m, new_v, step

    print(f"\n[Linear-Attention FT-Transformer] 학습 시작 - {n_epochs} epochs, batch={batch_size}")
    t_train_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch_start = time.perf_counter()

        for start in range(0, n, batch_size):
            idx     = perm[start : start + batch_size]
            y_batch = y_train[idx]
            tok_optimizer.zero_grad()

            # Step 1: each hospital tokenizes its own features locally.
            hospital_tokens = [tokenizers[i](X_trains[i][idx]) for i in range(NUM_CLIENTS)]
            # Step 2: concatenate into the fixed 8-token sequence.
            feature_tokens_t = torch.cat(hospital_tokens, dim=1)  # (batch, 8, d)

            # Step 3: top-model forward+backward (plain JAX for now).
            feature_tokens_j = jnp.array(feature_tokens_t.detach().numpy())
            loss, (grad_tokens_j, grad_params) = linear_attention_loss_and_grad(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)

            # Step 4: continue backward locally per hospital -- slice the
            # gradient back to each hospital's own token range.
            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            offset = 0
            for i in range(NUM_CLIENTS):
                k = len(SLEEP_FEATURE_GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

            epoch_loss += float(loss)
            n_batches  += 1

        epoch_wall_ms = (time.perf_counter() - t_epoch_start) * 1000

        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                test_tokens = torch.cat(
                    [tokenizers[i](X_tests[i]) for i in range(NUM_CLIENTS)], dim=1
                )
                logit = linear_attention_forward(jnp.array(test_tokens.numpy()), top_params)
                pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
                acc   = float((pred == y_test.numpy()).mean())
            print(f"  Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss/n_batches:.4f} | "
                  f"test_acc={acc:.4f} | wall={epoch_wall_ms:.1f}ms ({n_batches} batches)")
        else:
            print(f"  Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss/n_batches:.4f} | "
                  f"wall={epoch_wall_ms:.1f}ms ({n_batches} batches)")

    train_total_s = time.perf_counter() - t_train_start
    print(f"[Linear-Attention FT-Transformer] 학습 완료 — 총 {train_total_s:.1f}s")
    return tokenizers, top_params, scaler


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    run_linear_attention_simulation(csv_path, n_epochs=30, batch_size=32)
