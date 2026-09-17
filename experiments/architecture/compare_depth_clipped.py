"""Retry depth (num_layers=1,2,3) now that we know gradient clipping fixes
the phi(x)=x^2 blow-up (compare_width_clipping.py), and with a final
LayerNorm added before the classification head (linear_attention_deep_jax.py)
-- the earlier depth experiment failed catastrophically (loss in the
hundreds of millions) without either fix. d=32, matching the best width
found in compare_width_clipping.py. Clean 8-feature-only data.
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
from experiments.features.feature_ablation_experiment import load_with_features
from linear_attention_deep_jax import init_params_deep, forward_deep, loss_and_grad_deep, sigmoid_approx

BASE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]


def clip_global_norm(tree, max_norm):
    leaves = jax.tree_util.tree_leaves(tree)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, tree)


def tree_adam_init(params):
    return (jax.tree_util.tree_map(jnp.zeros_like, params), jax.tree_util.tree_map(jnp.zeros_like, params))


def tree_adam_update(params, grads, m, v, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    step += 1
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1 - beta2) * (g ** 2), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - beta1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - beta2 ** step), v)
    new_params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return new_params, m, v, step


def run_config(csv_path, num_layers, clip_norm, d=32, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    X, y, _ = load_with_features(csv_path, extra_features=[])
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=BASE_GROUPS
    )

    torch.manual_seed(seed)
    tokenizers = [FeatureTokenizer(len(BASE_GROUPS[i]), d) for i in range(NUM_CLIENTS)]
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

            loss, (grad_tokens_j, grad_params) = loss_and_grad_deep(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            if clip_norm is not None:
                grad_params = clip_global_norm(grad_params, clip_norm)
                grad_tokens_j = clip_global_norm(grad_tokens_j, clip_norm)

            top_params, m_state, v_state, step = tree_adam_update(top_params, grad_params, m_state, v_state, step, lr)

            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([p for tok in tokenizers for p in tok.parameters()], clip_norm)
            offset = 0
            for i in range(NUM_CLIENTS):
                k = len(BASE_GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

        with torch.no_grad():
            test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(NUM_CLIENTS)], dim=1)
            logit = forward_deep(jnp.array(test_tokens.numpy()), top_params)
            pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
            acc   = float((pred == y_test.numpy()).mean())
            best_acc = max(best_acc, acc)

    return acc, best_acc


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    print("=== depth x clipping (d=32, final LayerNorm added) ===\n")
    for num_layers in (1, 2, 3):
        for clip in (None, 1.0):
            t0 = time.perf_counter()
            final_acc, best_acc = run_config(csv_path, num_layers=num_layers, clip_norm=clip, d=32, n_epochs=30)
            tag = f"num_layers={num_layers}, clip={clip}"
            print(f"  {tag:26s} | final_acc={final_acc:.4f} | best_acc={best_acc:.4f} | {time.perf_counter()-t0:.1f}s")
