"""Isolate: does token-count normalization help or hurt the "+waso" (9-feature)
config specifically? Same data/split/training loop, only the top-model
forward differs (normalized vs the original un-normalized linear attention).
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from linear_attention_jax import init_params, sigmoid_approx, phi
from experiments.features.feature_ablation_experiment import load_with_features, BASE_GROUPS
from model import FeatureTokenizer
from sklearn.model_selection import train_test_split


def forward_unnormalized(feature_tokens, params):
    batch = feature_tokens.shape[0]
    cls = jnp.broadcast_to(params["cls_token"], (batch, 1, feature_tokens.shape[-1]))
    tokens = jnp.concatenate([cls, feature_tokens], axis=1)
    Q = tokens @ params["W_Q"]; K = tokens @ params["W_K"]; V = tokens @ params["W_V"]
    phi_Q, phi_K = phi(Q), phi(K)
    KV = jnp.einsum("bnd,bne->bde", phi_K, V)                # no /n_tokens
    attn_out = jnp.einsum("bnd,bde->bne", phi_Q, KV)
    cls_out = attn_out[:, 0, :]
    return cls_out @ params["W_head"] + params["b_head"]


def forward_normalized(feature_tokens, params):
    batch = feature_tokens.shape[0]
    cls = jnp.broadcast_to(params["cls_token"], (batch, 1, feature_tokens.shape[-1]))
    tokens = jnp.concatenate([cls, feature_tokens], axis=1)
    n_tokens = tokens.shape[1]
    Q = tokens @ params["W_Q"]; K = tokens @ params["W_K"]; V = tokens @ params["W_V"]
    phi_Q, phi_K = phi(Q), phi(K)
    KV = jnp.einsum("bnd,bne->bde", phi_K, V) / n_tokens
    attn_out = jnp.einsum("bnd,bde->bne", phi_Q, KV)
    cls_out = attn_out[:, 0, :]
    return cls_out @ params["W_head"] + params["b_head"]


def loss_fn(forward_fn, feature_tokens, y, params):
    logit = forward_fn(feature_tokens, params)
    pred = sigmoid_approx(logit)
    return jnp.mean((pred - y) ** 2)


def run(csv_path, forward_fn, extra_features, n_epochs=30, batch_size=32, d=16, lr=1e-3, seed=0):
    X, y, n_features = load_with_features(csv_path, extra_features)
    groups = [g[:] for g in BASE_GROUPS]
    next_idx = 8
    for _ in extra_features:
        groups[2].append(next_idx)
        next_idx += 1

    torch.manual_seed(seed)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizers = [FeatureTokenizer(len(groups[i]), d) for i in range(3)]
    tok_optimizer = torch.optim.Adam([p for tok in tokenizers for p in tok.parameters()], lr=lr)
    key = jax.random.PRNGKey(seed)
    top_params = init_params(key, d)
    m_state = {k: jnp.zeros_like(v) for k, v in top_params.items()}
    v_state = {k: jnp.zeros_like(v) for k, v in top_params.items()}
    beta1, beta2, adam_eps, step = 0.9, 0.999, 1e-8, 0

    loss_and_grad = jax.value_and_grad(lambda ft, y_, p: loss_fn(forward_fn, ft, y_, p), argnums=(0, 2))

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

    X_trains = [torch.tensor(X_train[:, g], dtype=torch.float32) for g in groups]
    X_tests  = [torch.tensor(X_test[:, g],  dtype=torch.float32) for g in groups]
    y_train  = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    y_test   = torch.tensor(y_test_raw,  dtype=torch.float32)
    n = len(y_train)

    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            y_batch = y_train[idx]
            tok_optimizer.zero_grad()
            hospital_tokens = [tokenizers[i](X_trains[i][idx]) for i in range(3)]
            feature_tokens_t = torch.cat(hospital_tokens, dim=1)
            feature_tokens_j = jnp.array(feature_tokens_t.detach().numpy())

            loss, (grad_tokens_j, grad_params) = loss_and_grad(feature_tokens_j, jnp.array(y_batch.numpy()), top_params)
            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)

            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            offset = 0
            for i in range(3):
                k = len(groups[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

    with torch.no_grad():
        test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(3)], dim=1)
        logit = forward_fn(jnp.array(test_tokens.numpy()), top_params)
        pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
        acc   = float((pred == y_test.numpy()).mean())
    return acc


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    print("=== +waso (9 features): normalized vs un-normalized linear attention ===")
    for name, fn in [("un-normalized (original)", forward_unnormalized), ("normalized (/n_tokens)", forward_normalized)]:
        t0 = time.perf_counter()
        acc = run(csv_path, fn, ["waso"], n_epochs=30)
        print(f"  {name:28s} | test_acc={acc:.4f} | {time.perf_counter()-t0:.1f}s")
