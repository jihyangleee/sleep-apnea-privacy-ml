"""8-feature baseline + waso + sleep_latency (confirmed helpful,
feature_ablation_experiment.py) + odi/mean_desat_duration/mean_desat_drop
(SpO2 desaturation-event features, extract_desat_features.py) -- all SpO2/
sleep-architecture derived, all plausibly watch-producible (unlike CVHR,
which needed ECG precision and was dropped -- a PPG-noise simulation
showed the CVHR index doesn't survive watch-grade pulse signals; those
scripts were removed, see README "제거된 실험").

Matched to whichever subjects have a downloaded annotations-events-nsrr
XML (currently ~1,324 of 5,804), same d=32/clip=1.0 config that gave the
best linear-attention result so far.
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
from extract_desat_features import extract_all

BASE_FEATURES = ["avgsat", "avg_hr", "slpprdp", "slpeffp", "times34p", "age_s1", "gender", "bmi_s1"]
EXTRA_FEATURES = ["waso", "sleep_latency"]
DESAT_FEATURES = ["odi", "mean_desat_duration", "mean_desat_drop"]
HR_COLS = ["savbnbh", "savbnoh", "savbrbh", "savbroh"]
LABEL = "ahi_a0h3a"
AHI_THRESHOLD = 15

FEATURES = BASE_FEATURES + EXTRA_FEATURES + DESAT_FEATURES  # 13 total
# desat (SpO2-derived) -> hospital A (already holds avgsat/avg_hr);
# waso/sleep_latency (PSG-lab sleep architecture) -> hospital B
GROUPS = [[0, 1, 10, 11, 12], [2, 3, 4, 8, 9], [5, 6, 7]]


def load_combined(csv_path, xml_dir):
    usecols = [c for c in BASE_FEATURES + EXTRA_FEATURES if c != "avg_hr"] + HR_COLS + [LABEL, "nsrrid"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["avg_hr"] = df[HR_COLS].mean(axis=1)
    df = df[df[LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna(subset=BASE_FEATURES + EXTRA_FEATURES + [LABEL, "nsrrid"])
    df = df.set_index("nsrrid")

    desat_df = extract_all(xml_dir)  # indexed by nsrrid
    df = df.join(desat_df, how="inner")  # matched subjects only (XML-downloaded)

    X = df[FEATURES].values.astype(np.float32)
    y = (df[LABEL].values >= AHI_THRESHOLD).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def run(csv_path, xml_dir, d=32, clip_norm=1.0, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    X, y = load_combined(csv_path, xml_dir)
    n_clients = len(GROUPS)
    torch.manual_seed(seed)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizers = [FeatureTokenizer(len(g), d) for g in GROUPS]
    tok_optimizer = torch.optim.Adam([p for tok in tokenizers for p in tok.parameters()], lr=lr)
    key = jax.random.PRNGKey(seed)
    top_params = init_params(key, d)
    m_state = jax.tree_util.tree_map(jnp.zeros_like, top_params)
    v_state = jax.tree_util.tree_map(jnp.zeros_like, top_params)
    beta1, beta2, adam_eps, step = 0.9, 0.999, 1e-8, 0

    def clip_global_norm(tree, max_norm):
        leaves = jax.tree_util.tree_leaves(tree)
        total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
        return jax.tree_util.tree_map(lambda g: g * scale, tree)

    def adam_update(params, grads, m, v, step):
        step += 1
        m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1 - beta2) * (g ** 2), v, grads)
        m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - beta1 ** step), m)
        v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - beta2 ** step), v)
        new_params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + adam_eps), params, m_hat, v_hat)
        return new_params, m, v, step

    X_trains = [torch.tensor(X_train[:, g], dtype=torch.float32) for g in GROUPS]
    X_tests = [torch.tensor(X_test[:, g], dtype=torch.float32) for g in GROUPS]
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
            hospital_tokens = [tokenizers[i](X_trains[i][idx]) for i in range(n_clients)]
            feature_tokens_t = torch.cat(hospital_tokens, dim=1)
            feature_tokens_j = jnp.array(feature_tokens_t.detach().numpy())
            loss, (grad_tokens_j, grad_params) = linear_attention_loss_and_grad(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            grad_params = clip_global_norm(grad_params, clip_norm)
            grad_tokens_j = clip_global_norm(grad_tokens_j, clip_norm)
            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)
            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            torch.nn.utils.clip_grad_norm_([p for tok in tokenizers for p in tok.parameters()], clip_norm)
            offset = 0
            for i in range(n_clients):
                k = len(GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

        with torch.no_grad():
            test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(n_clients)], dim=1)
            logit = linear_attention_forward(jnp.array(test_tokens.numpy()), top_params)
            pred = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
            acc = float((pred == y_test.numpy()).mean())
            best_acc = max(best_acc, acc)

    return acc, best_acc, len(y_train) + len(y_test)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    xml_dir = sys.argv[2] if len(sys.argv) > 2 else "shhs/polysomnography/annotations-events-nsrr/shhs1"
    t0 = time.perf_counter()
    final_acc, best_acc, n = run(csv_path, xml_dir)
    print(f"8-feature + waso + sleep_latency + desat(odi/dur/drop) | n={n} | "
          f"final_acc={final_acc:.4f} | best_acc={best_acc:.4f} | {time.perf_counter()-t0:.1f}s")
