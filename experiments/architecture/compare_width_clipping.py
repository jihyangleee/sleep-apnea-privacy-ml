"""8-feature baseline (unchanged features), varying token dim (d) and adding
gradient-norm clipping -- both depth-free levers, targeting the repeatedly
observed phi(x)=x^2 instability directly (clipping) or just adding capacity
without adding multiplicative depth (wider d).
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import partition_data_vertical
from model import FeatureTokenizer
from simulate import NUM_CLIENTS
from linear_attention_jax import init_params, linear_attention_forward, linear_attention_loss_and_grad, sigmoid_approx
from experiments.features.feature_ablation_experiment import load_with_features  # clean 8-feature-only loader,
# unaffected by dataset.py's SHHS_FEATURES (which now includes the 4 candidate columns)

BASE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]


def clip_global_norm(tree, max_norm):
    leaves = jax.tree_util.tree_leaves(tree)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, tree), float(total_norm)


def run_config(csv_path, d, clip_norm, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    X, y, n_features = load_with_features(csv_path, extra_features=[])  # clean 8-feature-only
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=BASE_GROUPS
    )

    torch.manual_seed(seed)
    tokenizers = [FeatureTokenizer(len(BASE_GROUPS[i]), d) for i in range(NUM_CLIENTS)]
    tok_optimizer = torch.optim.Adam([p for tok in tokenizers for p in tok.parameters()], lr=lr)
    key = jax.random.PRNGKey(seed)
    top_params = init_params(key, d)
    m_state = {k: jnp.zeros_like(v) for k, v in top_params.items()}
    v_state = {k: jnp.zeros_like(v) for k, v in top_params.items()}
    beta1, beta2, adam_eps, step = 0.9, 0.999, 1e-8, 0

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

    X_trains = [torch.tensor(partitions[i]["X_train"], dtype=torch.float32) for i in range(NUM_CLIENTS)]
    X_tests  = [torch.tensor(partitions[i]["X_test"],  dtype=torch.float32) for i in range(NUM_CLIENTS)]
    y_train  = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    y_test   = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)
    best_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            y_batch = y_train[idx]
            tok_optimizer.zero_grad()

            hospital_tokens = [tokenizers[i](X_trains[i][idx]) for i in range(NUM_CLIENTS)]
            feature_tokens_t = torch.cat(hospital_tokens, dim=1)
            feature_tokens_j = jnp.array(feature_tokens_t.detach().numpy())

            loss, (grad_tokens_j, grad_params) = linear_attention_loss_and_grad(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            if clip_norm is not None:
                grad_params, _ = clip_global_norm(grad_params, clip_norm)
                grad_tokens_j, _ = clip_global_norm(grad_tokens_j, clip_norm)

            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)

            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for tok in tokenizers for p in tok.parameters()], clip_norm
                )
            offset = 0
            for i in range(NUM_CLIENTS):
                k = len(BASE_GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

        with torch.no_grad():
            test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(NUM_CLIENTS)], dim=1)
            logit = linear_attention_forward(jnp.array(test_tokens.numpy()), top_params)
            pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
            acc   = float((pred == y_test.numpy()).mean())
            best_acc = max(best_acc, acc)

    return acc, best_acc


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    configs = [
        ("d=16, no clip (baseline)", 16, None),
        ("d=16, clip=1.0",           16, 1.0),
        ("d=32, no clip",            32, None),
        ("d=32, clip=1.0",           32, 1.0),
        ("d=64, no clip",            64, None),
        ("d=64, clip=1.0",           64, 1.0),
    ]
    print("=== 8 features unchanged: width (d) x gradient clipping ===\n")
    for name, d, clip in configs:
        t0 = time.perf_counter()
        final_acc, best_acc = run_config(csv_path, d=d, clip_norm=clip, n_epochs=30)
        print(f"  {name:26s} | final_epoch_acc={final_acc:.4f} | best_acc(any epoch)={best_acc:.4f} | {time.perf_counter()-t0:.1f}s")
