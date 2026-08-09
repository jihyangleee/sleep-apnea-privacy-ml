"""Empirical answer to "would stacking linear-attention layers help accuracy":
trains num_layers in {1, 2, 3} (pre-LN + residual, linear_attention_deep_jax.py)
on the same real SHHS data/split as simulate_linear_attention.py's single-block
baseline (73.3% test_acc), plain JAX, no HE/MPC involved -- purely to check
whether depth is worth the (separately unsolved) crypto cost before building it.
"""

import sys
import time
import numpy as np
import torch
import jax
import jax.numpy as jnp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import load_shhs_data, partition_data_vertical
from model import FeatureTokenizer
from simulate import SLEEP_FEATURE_GROUPS, NUM_CLIENTS
from linear_attention_deep_jax import init_params_deep, forward_deep, loss_and_grad_deep, sigmoid_approx


def tree_adam_init(params):
    return (
        jax.tree_util.tree_map(jnp.zeros_like, params),
        jax.tree_util.tree_map(jnp.zeros_like, params),
    )


def tree_adam_update(params, grads, m, v, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    step += 1
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1 - beta2) * (g ** 2), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - beta1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - beta2 ** step), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return new_params, m, v, step


def run_one_config(csv_path, num_layers, n_epochs=30, batch_size=32, d=16, lr=1e-3, seed=0):
    X, y, scaler = load_shhs_data(csv_path, return_scaler=True)
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=SLEEP_FEATURE_GROUPS
    )

    torch.manual_seed(seed)
    tokenizers = [FeatureTokenizer(len(SLEEP_FEATURE_GROUPS[i]), d) for i in range(NUM_CLIENTS)]
    tok_optimizer = torch.optim.Adam([p for tok in tokenizers for p in tok.parameters()], lr=lr)

    key = jax.random.PRNGKey(seed)
    top_params = init_params_deep(key, d, num_layers)
    m_state, v_state = tree_adam_init(top_params)
    step = 0

    X_trains = [torch.tensor(partitions[i]["X_train"], dtype=torch.float32) for i in range(NUM_CLIENTS)]
    X_tests  = [torch.tensor(partitions[i]["X_test"],  dtype=torch.float32) for i in range(NUM_CLIENTS)]
    y_train  = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    y_test   = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            y_batch = y_train[idx]
            tok_optimizer.zero_grad()

            hospital_tokens = [tokenizers[i](X_trains[i][idx]) for i in range(NUM_CLIENTS)]
            feature_tokens_t = torch.cat(hospital_tokens, dim=1)
            feature_tokens_j = jnp.array(feature_tokens_t.detach().numpy())

            loss, (grad_tokens_j, grad_params) = loss_and_grad_deep(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            top_params, m_state, v_state, step = tree_adam_update(
                top_params, grad_params, m_state, v_state, step, lr
            )

            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            offset = 0
            for i in range(NUM_CLIENTS):
                k = len(SLEEP_FEATURE_GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

            epoch_loss += float(loss)
            n_batches += 1

        if epoch % 10 == 0 or epoch == n_epochs or epoch == 1:
            with torch.no_grad():
                test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(NUM_CLIENTS)], dim=1)
                logit = forward_deep(jnp.array(test_tokens.numpy()), top_params)
                pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
                acc   = float((pred == y_test.numpy()).mean())
            print(f"    [num_layers={num_layers}] epoch {epoch:3d}/{n_epochs} | "
                  f"loss={epoch_loss/n_batches:.4f} | test_acc={acc:.4f}")

    wall_s = time.perf_counter() - t0
    return acc, wall_s


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    results = {}
    for num_layers in (1, 2, 3):
        print(f"\n=== num_layers={num_layers} (pre-LN + residual, phi(x)=x^2 linear attention) ===")
        acc, wall_s = run_one_config(csv_path, num_layers=num_layers, n_epochs=30)
        results[num_layers] = (acc, wall_s)

    print("\n=== Summary ===")
    print(f"  baseline (no LayerNorm, no residual, linear_attention_jax.py): test_acc=0.7331 (already measured)")
    for num_layers, (acc, wall_s) in results.items():
        print(f"  num_layers={num_layers} (with LayerNorm+residual): test_acc={acc:.4f}  ({wall_s:.1f}s)")
