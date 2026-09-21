"""Plaintext comparison on the 8 / 13 (desat) / 15 (deployed) feature sets:
logistic regression vs. 2nd-order logistic regression vs. low-degree polynomial
MLPs vs. gradient boosting (upper bound only -- not MPC/CKKS friendly).

Answers "is a hidden layer worth its CKKS-depth / MPC-round cost over plain
logistic regression?" -- so the MLPs use the same PolyActivation as model.py
and the same 3-hospital feature split (GROUPS).

    mlp_local   : model.py structure -- per-hospital Linear -> PolyAct -> Linear,
                  then a single linear top over the concatenated embeddings
                  (no cross-hospital interaction).
    mlp_top     : same, plus PolyAct -> Linear on the concatenated embeddings
                  (cross-hospital interaction; costs extra depth / MPC rounds).

Loss is BCE with a true sigmoid here (plaintext reference); the MPC path uses
MSE + polynomial sigmoid, so absolute numbers can differ slightly.
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset import load_shhs_data
from experiments.features.combined_desat_experiment import load_combined, GROUPS as DESAT_GROUPS
from experiments.features.feature_ablation_experiment import load_with_features, BASE_GROUPS
from model import PolyActivation
from simulate import SLEEP_FEATURE_GROUPS

CSV_PATH = "shhs/datasets/shhs1-dataset-0.21.0.csv"
XML_DIR = "shhs/polysomnography/annotations-events-nsrr/shhs1"
SEEDS = [0, 1, 2, 3, 4]


class PolyMLP(nn.Module):
    def __init__(self, groups, emb_dim=16, nonlinear_top=False):
        super().__init__()
        self.groups = groups
        self.subs = nn.ModuleList(
            nn.Sequential(nn.Linear(len(g), emb_dim), PolyActivation(), nn.Linear(emb_dim, emb_dim))
            for g in groups
        )
        total = emb_dim * len(groups)
        if nonlinear_top:
            self.top = nn.Sequential(nn.Linear(total, emb_dim), PolyActivation(), nn.Linear(emb_dim, 1))
        else:
            self.top = nn.Linear(total, 1)

    def forward(self, x):
        embs = [sub(x[:, g]) for sub, g in zip(self.subs, self.groups)]
        return self.top(torch.cat(embs, dim=1)).squeeze(1)


def train_mlp(X_tr, y_tr, X_te, seed, nonlinear_top, groups, n_epochs=60, batch=64, lr=3e-3, clip=1.0):
    torch.manual_seed(seed)
    model = PolyMLP(groups, nonlinear_top=nonlinear_top)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    n = len(yt)
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    with torch.no_grad():
        logit = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def fit_predict(name, X_tr, y_tr, X_te, seed, groups):
    if name == "logreg":
        return LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    if name == "logreg_poly2":
        pf = PolynomialFeatures(2, include_bias=False)
        clf = LogisticRegression(max_iter=5000, C=0.5).fit(pf.fit_transform(X_tr), y_tr)
        return clf.predict_proba(pf.transform(X_te))[:, 1]
    if name == "mlp_local":
        return train_mlp(X_tr, y_tr, X_te, seed, False, groups)
    if name == "mlp_top":
        return train_mlp(X_tr, y_tr, X_te, seed, True, groups)
    if name == "hist_gbdt (upper bound)":
        return HistGradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    raise ValueError(name)


MODELS = ["logreg", "logreg_poly2", "mlp_local", "mlp_top", "hist_gbdt (upper bound)"]

# name -> (loader returning X, y; hospital feature-index groups)
CONFIGS = {
    "8feat":  (lambda: load_with_features(CSV_PATH, [])[:2], BASE_GROUPS),
    "15feat": (lambda: load_shhs_data(CSV_PATH), SLEEP_FEATURE_GROUPS),  # deployed: dataset.MODEL_FEATURES (12 + desat 3)
    "13feat": (lambda: load_combined(CSV_PATH, XML_DIR), DESAT_GROUPS),  # + desat, XML-matched subjects only
}


def run_config(cfg_name):
    loader, groups = CONFIGS[cfg_name]
    X, y = loader()
    print(f"\n##### {cfg_name} | n={len(y)} | features={X.shape[1]} | AHI>=15 prevalence={y.mean():.3f}")
    results = {m: {"acc": [], "auc": [], "rec": []} for m in MODELS}

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        for m in MODELS:
            p = fit_predict(m, X_tr, y_tr, X_te, seed, groups)
            pred = (p > 0.5).astype(np.float32)
            results[m]["acc"].append(float((pred == y_te).mean()))
            results[m]["auc"].append(float(roc_auc_score(y_te, p)))
            results[m]["rec"].append(float(recall_score(y_te, pred)))

    print(f"=== {cfg_name}: mean ± std over {len(SEEDS)} seeds (different stratified splits) ===")
    print(f"{'model':28s} {'acc':>15s} {'auc':>15s} {'recall':>15s}")
    for m in MODELS:
        r = results[m]
        print(f"{m:28s} {np.mean(r['acc']):.4f}±{np.std(r['acc']):.4f}  "
              f"{np.mean(r['auc']):.4f}±{np.std(r['auc']):.4f}  "
              f"{np.mean(r['rec']):.4f}±{np.std(r['rec']):.4f}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    for name in names:
        run_config(name)
