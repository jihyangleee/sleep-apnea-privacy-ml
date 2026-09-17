"""Plaintext-only ceiling check: does the MSE + cubic-sigmoid-approx loss
(chosen because BCE's gradient needs a secure-division protocol under
MPC/HE -- see submodel.md, linear_attention_jax.py) actually cost accuracy
vs a standard BCE + true-sigmoid loss? Same phi(x)=x^2 linear-attention
architecture, same data/split, only the loss/output nonlinearity differs.
Ignores HE/MPC cost entirely -- purely to check whether solving the
secure-BCE-gradient problem would even be worth it before attempting it,
same rationale as compare_depth_experiment.py.

Also carries gradient-norm clipping (clip=1.0) and d=32, since
compare_width_clipping.py found that combination beats the un-clipped
d=16 baseline substantially (0.6898 -> 0.7368).
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

BASE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]


def init_params(key, d):
    k_cls, k_q, k_k, k_v, k_head = jax.random.split(key, 5)
    scale = 1.0 / jnp.sqrt(d)
    return {
        "cls_token": jax.random.normal(k_cls, (d,)) * scale,
        "W_Q": jax.random.normal(k_q, (d, d)) * scale,
        "W_K": jax.random.normal(k_k, (d, d)) * scale,
        "W_V": jax.random.normal(k_v, (d, d)) * scale,
        "W_head": jax.random.normal(k_head, (d, 1)) * scale,
        "b_head": jnp.zeros((1,)),
    }


def phi(x):
    return x ** 2


def forward(tokens_feat, params):
    batch = tokens_feat.shape[0]
    cls = jnp.broadcast_to(params["cls_token"], (batch, 1, tokens_feat.shape[-1]))
    tokens = jnp.concatenate([cls, tokens_feat], axis=1)
    Q = tokens @ params["W_Q"]; K = tokens @ params["W_K"]; V = tokens @ params["W_V"]
    n_tokens = tokens.shape[1]
    phi_Q, phi_K = phi(Q), phi(K)
    KV = jnp.einsum("bnd,bne->bde", phi_K, V) / n_tokens
    attn_out = jnp.einsum("bnd,bde->bne", phi_Q, KV)
    cls_out = attn_out[:, 0, :]
    return cls_out @ params["W_head"] + params["b_head"]


def sigmoid_approx(logit):
    return 0.5 + 0.197 * logit - 0.004 * logit ** 3


def loss_mse(tokens_feat, y, params):
    pred = sigmoid_approx(forward(tokens_feat, params))
    return jnp.mean((pred - y) ** 2)


def loss_bce(tokens_feat, y, params):
    logit = forward(tokens_feat, params)
    # standard numerically-stable BCE-with-logits (true sigmoid, plaintext only)
    return jnp.mean(jnp.maximum(logit, 0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit))))


loss_and_grad_mse = jax.value_and_grad(loss_mse, argnums=(0, 2))
loss_and_grad_bce = jax.value_and_grad(loss_bce, argnums=(0, 2))


def clip_global_norm(tree, max_norm):
    leaves = jax.tree_util.tree_leaves(tree)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, tree)


def run_config(csv_path, loss_name, d=32, clip_norm=1.0, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    X, y, _ = load_with_features(csv_path, extra_features=[])
    partitions, _, _, y_train_raw, y_test_raw = partition_data_vertical(
        X, y, num_clients=NUM_CLIENTS, feature_groups=BASE_GROUPS
    )

    torch.manual_seed(seed)
    tokenizers = [FeatureTokenizer(len(BASE_GROUPS[i]), d) for i in range(NUM_CLIENTS)]
    tok_optimizer = torch.optim.Adam([p for tok in tokenizers for p in tok.parameters()], lr=lr)

    key = jax.random.PRNGKey(seed)
    top_params = init_params(key, d)
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

    loss_and_grad = loss_and_grad_bce if loss_name == "bce" else loss_and_grad_mse

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

            loss, (grad_tokens_j, grad_params) = loss_and_grad(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            if clip_norm is not None:
                grad_params = clip_global_norm(grad_params, clip_norm)
                grad_tokens_j = clip_global_norm(grad_tokens_j, clip_norm)
            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)

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
            logit = forward(jnp.array(test_tokens.numpy()), top_params)
            pred_prob = sigmoid_approx(logit) if loss_name == "mse" else jax.nn.sigmoid(logit)
            pred = (np.array(pred_prob).squeeze() > 0.5).astype(np.float32)
            acc = float((pred == y_test.numpy()).mean())
            best_acc = max(best_acc, acc)

    return acc, best_acc


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    print("=== MSE+cubic-sigmoid-approx (HE-compatible) vs BCE+true-sigmoid (plaintext-only ceiling), d=32, clip=1.0 ===\n")
    for name, loss_name in [("MSE + sigmoid_approx (current)", "mse"), ("BCE + true sigmoid (plaintext ceiling)", "bce")]:
        t0 = time.perf_counter()
        final_acc, best_acc = run_config(csv_path, loss_name)
        print(f"  {name:42s} | final_acc={final_acc:.4f} | best_acc={best_acc:.4f} | {time.perf_counter()-t0:.1f}s")
