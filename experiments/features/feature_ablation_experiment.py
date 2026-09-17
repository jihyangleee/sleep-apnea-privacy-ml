"""Which of the 4 candidate features (waso, sleep_latency, neck20, ess_s1)
actually helps vs. hurts, added one at a time to the 8-feature baseline?
Uses the token-count-normalized linear_attention_jax.py (already fixed for
the N-token scale issue) so any remaining accuracy gap reflects the
feature's actual signal, not training instability.
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model import FeatureTokenizer
from linear_attention_jax import init_params, linear_attention_forward, linear_attention_loss_and_grad, sigmoid_approx

BASE_FEATURES = ["avgsat", "avg_hr", "slpprdp", "slpeffp", "times34p", "age_s1", "gender", "bmi_s1"]
CANDIDATES    = ["waso", "sleep_latency", "neck20", "ess_s1"]
HR_COLS       = ["savbnbh", "savbnoh", "savbrbh", "savbroh"]
LABEL         = "ahi_a0h3a"
AHI_THRESHOLD = 15

# base hospital split (feature-index groups get rebuilt per-config below)
BASE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]


def load_with_features(csv_path, extra_features):
    features = BASE_FEATURES + extra_features
    usecols  = [c for c in features if c != "avg_hr"] + HR_COLS + [LABEL]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["avg_hr"] = df[HR_COLS].mean(axis=1)
    df = df[df[LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna()
    X = df[features].values.astype(np.float32)
    y = (df[LABEL].values >= AHI_THRESHOLD).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, len(features)


def run_config(csv_path, extra_features, n_epochs=30, batch_size=32, d=16, lr=1e-3, seed=0):
    X, y, n_features = load_with_features(csv_path, extra_features)
    groups = [g[:] for g in BASE_GROUPS]
    # new feature(s) go to hospital C (clinic-type group, index 2), matching
    # the earlier assignment rationale (neck/ess are clinic/questionnaire-like;
    # waso/sleep_latency are sleep-lab-like -- but for this ablation we just
    # need *a* consistent placement per feature, hospital C is fine for all).
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

            loss, (grad_tokens_j, grad_params) = linear_attention_loss_and_grad(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
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
        logit = linear_attention_forward(jnp.array(test_tokens.numpy()), top_params)
        pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
        acc   = float((pred == y_test.numpy()).mean())
    return acc, n_features, len(y)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"

    configs = [("baseline (8 features)", [])] + [
        (f"+{feat}", [feat]) for feat in CANDIDATES
    ] + [
        ("+waso+sleep_latency (10 features)", ["waso", "sleep_latency"]),
    ]

    print("=== Feature ablation: 8-feature baseline vs each candidate added individually ===")
    print("(token-count-normalized linear attention, 30 epochs each, same seed/split logic)\n")
    for name, extra in configs:
        t0 = time.perf_counter()
        acc, n_feat, n_samples = run_config(csv_path, extra, n_epochs=30)
        wall = time.perf_counter() - t0
        print(f"  {name:28s} | n_features={n_feat:2d} | n_samples={n_samples:5d} | "
              f"test_acc={acc:.4f} | {wall:.1f}s")
