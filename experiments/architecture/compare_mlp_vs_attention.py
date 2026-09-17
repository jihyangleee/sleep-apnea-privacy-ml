"""Does linear attention actually earn its complexity over a plain MLP top
model, at the same d=32/clip=1.0 setting that gave the best linear-attention
result so far (0.7368 final / 0.7406 best, compare_width_clipping.py)?

Same FeatureTokenizer front-end (fair comparison -- same embedding
capacity per hospital), only the top model differs:
  - attention: phi(Q)@(phi(K)^T@V), phi(x)=x^2      (linear_attention_jax.py)
  - mlp:       concat tokens -> Linear -> x^2 -> Linear -> logit
Both are single-hidden-nonlinearity, HE-comparable-depth top models (the
MLP's x^2 activation is the same "1 ciphertext-squaring level" cost as
attention's phi(Q)/phi(K), just applied once instead of twice, so if
anything the MLP is slightly *cheaper* under HE, not more expensive).
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
from linear_attention_jax import init_params as init_attn_params, linear_attention_forward, \
    linear_attention_loss_and_grad, sigmoid_approx

BASE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]
N_TOKENS = 9  # CLS + 8 features


def init_mlp_params(key, d, n_tokens=N_TOKENS):
    k1, k2 = jax.random.split(key, 2)
    scale1 = 1.0 / jnp.sqrt(n_tokens * d)
    scale2 = 1.0 / jnp.sqrt(d)
    return {
        "cls_token": jax.random.normal(jax.random.fold_in(key, 0), (d,)) * (1.0 / jnp.sqrt(d)),
        "W1": jax.random.normal(k1, (n_tokens * d, d)) * scale1,
        "b1": jnp.zeros((d,)),
        "W2": jax.random.normal(k2, (d, 1)) * scale2,
        "b2": jnp.zeros((1,)),
    }


def mlp_forward(tokens_feat, params):
    batch = tokens_feat.shape[0]
    cls = jnp.broadcast_to(params["cls_token"], (batch, 1, tokens_feat.shape[-1]))
    tokens = jnp.concatenate([cls, tokens_feat], axis=1)  # (batch, n_tokens, d)
    flat = tokens.reshape(batch, -1)
    h = flat @ params["W1"] + params["b1"]
    h = h ** 2  # same phi(x)=x^2 nonlinearity family as the attention model
    return h @ params["W2"] + params["b2"]


def mlp_loss(tokens_feat, y, params):
    pred = sigmoid_approx(mlp_forward(tokens_feat, params))
    return jnp.mean((pred - y) ** 2)


mlp_loss_and_grad = jax.value_and_grad(mlp_loss, argnums=(0, 2))


def clip_global_norm(tree, max_norm):
    leaves = jax.tree_util.tree_leaves(tree)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, tree)


def run_config(csv_path, model_name, d=32, clip_norm=1.0, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    X, y, _ = load_with_features(csv_path, extra_features=[])
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=BASE_GROUPS
    )

    torch.manual_seed(seed)
    tokenizers = [FeatureTokenizer(len(BASE_GROUPS[i]), d) for i in range(NUM_CLIENTS)]
    tok_optimizer = torch.optim.Adam([p for tok in tokenizers for p in tok.parameters()], lr=lr)

    key = jax.random.PRNGKey(seed)
    if model_name == "attention":
        top_params = init_attn_params(key, d)
        forward_fn, loss_and_grad_fn = linear_attention_forward, linear_attention_loss_and_grad
    else:
        top_params = init_mlp_params(key, d)
        forward_fn, loss_and_grad_fn = mlp_forward, mlp_loss_and_grad

    m_state = jax.tree_util.tree_map(jnp.zeros_like, top_params)
    v_state = jax.tree_util.tree_map(jnp.zeros_like, top_params)
    beta1, beta2, adam_eps, step = 0.9, 0.999, 1e-8, 0

    def adam_update(params, grads, m, v, step):
        step += 1
        m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1 - beta2) * (g ** 2), v, grads)
        m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - beta1 ** step), m)
        v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - beta2 ** step), v)
        new_params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + adam_eps), params, m_hat, v_hat)
        return new_params, m, v, step

    X_trains = [torch.tensor(partitions[i]["X_train"], dtype=torch.float32) for i in range(NUM_CLIENTS)]
    X_tests = [torch.tensor(partitions[i]["X_test"], dtype=torch.float32) for i in range(NUM_CLIENTS)]
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32)
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

            loss, (grad_tokens_j, grad_params) = loss_and_grad_fn(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            grad_params = clip_global_norm(grad_params, clip_norm)
            grad_tokens_j = clip_global_norm(grad_tokens_j, clip_norm)
            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)

            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            torch.nn.utils.clip_grad_norm_([p for tok in tokenizers for p in tok.parameters()], clip_norm)
            offset = 0
            for i in range(NUM_CLIENTS):
                k = len(BASE_GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

        with torch.no_grad():
            test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(NUM_CLIENTS)], dim=1)
            logit = forward_fn(jnp.array(test_tokens.numpy()), top_params)
            pred = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
            acc = float((pred == y_test.numpy()).mean())
            best_acc = max(best_acc, acc)

    return acc, best_acc


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    print("=== linear attention vs plain MLP top model, d=32, clip=1.0, same tokenizer ===\n")
    for name in ["attention", "mlp"]:
        t0 = time.perf_counter()
        final_acc, best_acc = run_config(csv_path, name)
        print(f"  {name:10s} | final_acc={final_acc:.4f} | best_acc={best_acc:.4f} | {time.perf_counter()-t0:.1f}s")
