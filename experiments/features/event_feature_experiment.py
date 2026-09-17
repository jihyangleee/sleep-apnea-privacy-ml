"""Does adding SpO2-desaturation-event features (ODI, mean duration, mean
drop -- parsed from NSRR SHHS1 annotation XML by extract_desat_features.py)
beat the 8-feature summary-statistic baseline (test_acc=0.7331, see
compare_depth_experiment.py)?

Only the desaturation-event subset of subjects (those with a downloaded and
parsed annotations-events-nsrr/shhs1 XML) can be used, so the baseline is
RE-TRAINED on that same matched subset rather than compared against the old
0.7331 number directly -- otherwise a sample-size difference would confound
the comparison. Same architecture (single-block linear_attention_jax.py),
same split/seed/epochs as feature_ablation_experiment.py so the only
difference between the two configs is the 3 extra columns.
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
DESAT_FEATURES = ["odi", "mean_desat_duration", "mean_desat_drop"]
HR_COLS   = ["savbnbh", "savbnoh", "savbrbh", "savbroh"]
LABEL     = "ahi_a0h3a"
AHI_THRESHOLD = 15

# hospital A already holds avgsat/avg_hr (SpO2 + HR monitoring) -- desat
# features are SpO2-derived too, so they join hospital A's group rather than
# creating a 4th party or going to an unrelated hospital.
BASE_GROUPS      = [[0, 1], [2, 3, 4], [5, 6, 7]]
WITH_DESAT_GROUPS = [[0, 1, 8, 9, 10], [2, 3, 4], [5, 6, 7]]


def load_matched(csv_path, xml_dir, with_desat: bool):
    usecols = [c for c in BASE_FEATURES if c != "avg_hr"] + HR_COLS + [LABEL, "nsrrid"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["avg_hr"] = df[HR_COLS].mean(axis=1)
    df = df[df[LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna(subset=BASE_FEATURES + [LABEL, "nsrrid"])

    desat_df = extract_all(xml_dir)  # indexed by nsrrid
    df = df.set_index("nsrrid").join(desat_df, how="inner")  # matched subjects only

    features = BASE_FEATURES + (DESAT_FEATURES if with_desat else [])
    X = df[features].values.astype(np.float32)
    y = (df[LABEL].values >= AHI_THRESHOLD).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def run_config(csv_path, xml_dir, with_desat, n_epochs=30, batch_size=32, d=16, lr=1e-3, seed=0):
    X, y = load_matched(csv_path, xml_dir, with_desat)
    groups = WITH_DESAT_GROUPS if with_desat else BASE_GROUPS
    n_clients = len(groups)

    torch.manual_seed(seed)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizers = [FeatureTokenizer(len(groups[i]), d) for i in range(n_clients)]
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

            hospital_tokens = [tokenizers[i](X_trains[i][idx]) for i in range(n_clients)]
            feature_tokens_t = torch.cat(hospital_tokens, dim=1)
            feature_tokens_j = jnp.array(feature_tokens_t.detach().numpy())

            loss, (grad_tokens_j, grad_params) = linear_attention_loss_and_grad(
                feature_tokens_j, jnp.array(y_batch.numpy()), top_params
            )
            top_params, m_state, v_state, step = adam_update(top_params, grad_params, m_state, v_state, step)

            grad_tokens_t = torch.tensor(np.array(grad_tokens_j), dtype=torch.float32)
            offset = 0
            for i in range(n_clients):
                k = len(groups[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

    with torch.no_grad():
        test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(n_clients)], dim=1)
        logit = linear_attention_forward(jnp.array(test_tokens.numpy()), top_params)
        pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
        acc   = float((pred == y_test.numpy()).mean())
    return acc, len(y)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    xml_dir  = sys.argv[2] if len(sys.argv) > 2 else "shhs/polysomnography/annotations-events-nsrr/shhs1"

    print("=== 8-feature baseline vs. +SpO2-desaturation-event features (matched subjects only) ===\n")
    for name, with_desat in [("baseline (8 features, matched subset)", False),
                              ("+odi/mean_duration/mean_drop (11 features)", True)]:
        t0 = time.perf_counter()
        acc, n_samples = run_config(csv_path, xml_dir, with_desat, n_epochs=30)
        wall = time.perf_counter() - t0
        print(f"  {name:44s} | n_samples={n_samples:5d} | test_acc={acc:.4f} | {wall:.1f}s")
