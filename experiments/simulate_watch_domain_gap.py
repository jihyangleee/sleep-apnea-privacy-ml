"""Simulated PSG-vs-watch domain-gap check.

There is no real Galaxy Watch data anywhere in this repo (dataset.py's
_GALAXY_WATCH_PROFILES are hand-typed synthetic examples, and client_gui.py's
feature_processor.py import -- which would parse a real watch export -- does
not exist). SHHS participants never wore a Galaxy Watch, so there is no
paired PSG-vs-watch ground truth to validate against directly.

This script instead builds a literature-informed NOISE MODEL approximating
what a wrist-worn consumer wearable (specifically Galaxy Watch where a
device-specific number was available) would measure vs. the PSG ground
truth already in SHHS, then asks: does the model trained on clean PSG
features still classify AHI>=15 correctly when the *test-time* input looks
like noisy watch data instead of PSG?

This is an APPROXIMATION, not a real validation -- it tells you how
sensitive the model is to the kind/scale of noise these papers report, not
what a real deployed Galaxy Watch would actually produce.

Noise sources (bias, std added to RAW pre-scaling units), per feature:

  avgsat (SpO2, %)      bias=-0.2  std=2.3   Galaxy Watch 4 nocturnal SpO2 vs
                                              PSG, RMSE 2.3%, bias -0.2%.
                                              Cho et al., PMC11367728.
  avg_hr (bpm)           bias=0.0   std=3.0   Samsung smartwatch vs ECG during
                                              sleep (low-motion, best case for
                                              PPG): "acceptable accuracy,
                                              low error variance" -- no exact
                                              bpm MAE published in abstract,
                                              3 bpm is a conservative estimate
                                              from the reported low-error
                                              regime. PLOS ONE 2022,
                                              PMC9731465.
  slpprdp (TST, min)     bias=-16.85 std=30   Consumer wrist trackers vs PSG,
                                              24-study meta-analysis mean
                                              diff -16.85 min; std is this
                                              script's estimate of the
                                              between-study spread (not
                                              itself pooled in the abstract).
                                              JCSM meta-analysis, PMC11874098.
  slpeffp (%)             bias=-4.69  std=8    Same meta-analysis, mean diff
                                              -4.691 pp sleep efficiency.
  times34p (deep sleep %) bias=0.0    std=8    N3 is the worst-performing
                                              stage for consumer wearables --
                                              Withings Scanwatch N3+REM epoch
                                              accuracy 66.74%, Fitbit N3
                                              sensitivity >=0.50 only (~coin
                                              flip). No usable bias/std in
                                              physical units was published,
                                              so std=8pp is this script's
                                              rough stand-in for "far noisier
                                              than the other features",
                                              scaled relative to typical
                                              times34p values (~15-20%).
                                              academic.oup.com/sleepadvances
                                              zpaf021; PMC10654909.
  age_s1 / gender / bmi_s1              unaffected -- app-entered, same
                                              source at train and inference.

Everything above is a rough, literature-anchored approximation, not a
device-specific calibration -- treat the resulting accuracy as "how fragile
is this model to noise of roughly this shape", not a real product number.
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
from dataset import _SHHS_HR_COLS, SHHS_LABEL, AHI_THRESHOLD

# Clean 8-feature-only list (NOT dataset.py's SHHS_FEATURES, which now also
# includes waso/sleep_latency/neck20/ess_s1 -- those have heavy missingness
# and dropna() on them shrinks n=5000+ down to ~2500, which is not
# comparable to the established 0.7331 baseline). Matches
# feature_ablation_experiment.py's BASE_FEATURES.
SHHS_FEATURES = ["avgsat", "avg_hr", "slpprdp", "slpeffp", "times34p", "age_s1", "gender", "bmi_s1"]
BASE_GROUPS = [[0, 1], [2, 3, 4], [5, 6, 7]]

# (bias, std, clip_low, clip_high) in raw units; index-aligned to SHHS_FEATURES[:5]
NOISE_MODEL = {
    "avgsat":   (-0.2,  2.3,  70.0, 100.0),
    "avg_hr":   (0.0,   3.0,  30.0, 200.0),
    "slpprdp":  (-16.85, 30.0, 0.0, 700.0),
    "slpeffp":  (-4.69,  8.0,  0.0, 100.0),
    "times34p": (0.0,    8.0,  0.0, 60.0),
}


def load_raw(csv_path):
    csv_cols = [c for c in SHHS_FEATURES if c != "avg_hr"] + _SHHS_HR_COLS + [SHHS_LABEL]
    df = pd.read_csv(csv_path, usecols=csv_cols)
    df["avg_hr"] = df[_SHHS_HR_COLS].mean(axis=1)
    df = df[df[SHHS_LABEL] >= 0]
    df = df[df["slpeffp"] > 0]
    df = df.dropna()
    X_raw = df[SHHS_FEATURES].values.astype(np.float32)
    y = (df[SHHS_LABEL].values >= AHI_THRESHOLD).astype(np.float32)
    return X_raw, y


def apply_watch_noise(X_raw, rng):
    X_noisy = X_raw.copy()
    for i, name in enumerate(SHHS_FEATURES[:5]):
        bias, std, lo, hi = NOISE_MODEL[name]
        noise = rng.normal(bias, std, size=X_noisy.shape[0])
        X_noisy[:, i] = np.clip(X_noisy[:, i] + noise, lo, hi)
    return X_noisy


def train_model(X_train_scaled, y_train_raw, d=16, n_epochs=30, batch_size=32, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    tokenizers = [FeatureTokenizer(len(g), d) for g in BASE_GROUPS]
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

    X_trains = [torch.tensor(X_train_scaled[:, g], dtype=torch.float32) for g in BASE_GROUPS]
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
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
                k = len(BASE_GROUPS[i])
                hospital_tokens[i].backward(grad_tokens_t[:, offset:offset + k, :])
                offset += k
            tok_optimizer.step()

    return tokenizers, top_params


def evaluate(tokenizers, top_params, X_scaled, y_raw):
    X_tests = [torch.tensor(X_scaled[:, g], dtype=torch.float32) for g in BASE_GROUPS]
    with torch.no_grad():
        test_tokens = torch.cat([tokenizers[i](X_tests[i]) for i in range(3)], dim=1)
        logit = linear_attention_forward(jnp.array(test_tokens.numpy()), top_params)
        pred  = (np.array(sigmoid_approx(logit)).squeeze() > 0.5).astype(np.float32)
    return float((pred == y_raw).mean())


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "shhs/datasets/shhs1-dataset-0.21.0.csv"
    n_noise_repeats = 5

    X_raw, y = load_raw(csv_path)
    print(f"[data] n={len(y)}  AHI>=15 rate={y.mean():.1%}")

    scaler = StandardScaler()
    scaler.fit(X_raw)  # fit on all raw data, matching this repo's existing convention

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_raw[idx_train])
    X_test_scaled_clean = scaler.transform(X_raw[idx_test])
    y_train, y_test = y[idx_train], y[idx_test]

    t0 = time.perf_counter()
    tokenizers, top_params = train_model(X_train_scaled, y_train, n_epochs=30)
    print(f"[train] done in {time.perf_counter()-t0:.1f}s (PSG-only, clean features)")

    acc_clean = evaluate(tokenizers, top_params, X_test_scaled_clean, y_test)
    print(f"\n  clean PSG test_acc            = {acc_clean:.4f}  (n={len(y_test)})")

    noisy_accs = []
    rng = np.random.default_rng(0)
    for r in range(n_noise_repeats):
        X_test_noisy_raw = apply_watch_noise(X_raw[idx_test], rng)
        X_test_noisy_scaled = scaler.transform(X_test_noisy_raw)
        acc_noisy = evaluate(tokenizers, top_params, X_test_noisy_scaled, y_test)
        noisy_accs.append(acc_noisy)
        print(f"  simulated-watch test_acc [{r}]  = {acc_noisy:.4f}")

    noisy_accs = np.array(noisy_accs)
    print(f"\n  simulated-watch test_acc (mean +/- std over {n_noise_repeats} noise draws) "
          f"= {noisy_accs.mean():.4f} +/- {noisy_accs.std():.4f}")
    print(f"  domain-gap drop = {acc_clean - noisy_accs.mean():.4f}")
