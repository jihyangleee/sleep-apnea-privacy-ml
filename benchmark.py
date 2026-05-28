"""
python benchmark.py [--n-infer N] [--n-plain N] [--n-batches N] [--batch-size N]

Latency benchmark for the VFL + HE sleep apnea system.
  1. HE inference    : encrypt + server compute + decrypt per Galaxy Watch request
  2. Plaintext infer : same model, no HE
  3. SS+DH overhead  : embedding secret-sharing vs. plain concatenation per training batch
"""

import argparse
import time
import numpy as np
import torch
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from model import VerticalHeartNet
from he_client import HEInference, build_he_context
from secret_sharing import additive_split, DHMasker
from simulate import NUM_CLIENTS, EMB_DIM

MODEL_PATH = "vertical_model.pt"


# -- CLI args ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="VFL+HE latency benchmark")
parser.add_argument("--n-infer",    type=int, default=20,   help="HE inference requests (default 20)")
parser.add_argument("--n-plain",    type=int, default=1000, help="plaintext inference requests (default 1000)")
parser.add_argument("--n-batches",  type=int, default=50,   help="SS overhead batches (default 50)")
parser.add_argument("--batch-size", type=int, default=32,   help="training batch size (default 32)")
args = parser.parse_args()

N_INFER    = args.n_infer
N_PLAIN    = args.n_plain
N_BATCHES  = args.n_batches
BATCH_SIZE = args.batch_size


# -- helpers -------------------------------------------------------------------
W = 66

def banner(title):
    pad = (W - len(title) - 2) // 2
    print("\n" + "=" * W)
    print(" " * pad + f" {title} ")
    print("=" * W)

def section(title):
    print(f"\n  [{title}]")
    print("  " + "-" * (W - 2))

def row(label, value, unit="", indent=4):
    print(f"{' '*indent}{label:<36}{value}  {unit}")

def bar(ratio, width=20):
    filled = round(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# -- load model ----------------------------------------------------------------
banner("VFL + HE Latency Benchmark")
print(f"  Model      : {MODEL_PATH}")
print(f"  HE samples : {N_INFER}   Plain samples : {N_PLAIN}   SS batches : {N_BATCHES} x {BATCH_SIZE}")

ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = VerticalHeartNet(ckpt["feature_groups"], ckpt["emb_dim"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scaler = StandardScaler()
scaler.mean_          = np.array(ckpt["scaler_mean"])
scaler.scale_         = np.array(ckpt["scaler_scale"])
scaler.n_features_in_ = len(scaler.mean_)

# Generate synthetic Galaxy Watch users, normalised with the trained model's scaler
rng = np.random.default_rng(0)
N   = N_INFER + N_PLAIN
raw = np.column_stack([
    rng.normal(95.0,  2.5,  N).clip(70, 100),
    rng.normal(65.0,  10.0, N).clip(40, 120),
    rng.normal(380.0, 70.0, N).clip(60, 600),
    rng.normal(85.0,  10.0, N).clip(20, 100),
    rng.normal(15.0,  8.0,  N).clip(0,  50),
    rng.uniform(30,   75,   N),
    rng.choice([1.0, 2.0],  N),
    rng.normal(28.0,  6.0,  N).clip(15, 55),
]).astype(np.float32)
X_all   = scaler.transform(raw).astype(np.float32)
X_he    = X_all[:N_INFER]
X_plain = X_all[:N_PLAIN]


# -----------------------------------------------------------------------------
# 1. HE Inference
# -----------------------------------------------------------------------------
banner("1 / 3  HE Inference  (CKKS encrypted)")

print(f"\n  Building HE context ... ", end="", flush=True)
t0 = time.perf_counter()
ctx_patient = build_he_context()
ctx_server  = ts.context_from(ctx_patient.serialize(save_secret_key=False))
print(f"{(time.perf_counter()-t0)*1e3:.0f} ms  (one-time setup)")

he_module = HEInference(model)

# warmup
_enc = HEInference.encrypt_input(X_he[0], ctx_patient)
HEInference.decrypt_result(he_module.run_he_inference(_enc, ctx_server), ctx_patient)

t_enc, t_srv, t_dec = [], [], []

section(f"Per-request breakdown  (n={N_INFER})")
print(f"  {'#':>4}  {'encrypt':>10}  {'server':>10}  {'decrypt':>10}  {'total':>10}  p")
print("  " + "-" * (W - 2))

for i, x in enumerate(X_he):
    t0 = time.perf_counter()
    enc = HEInference.encrypt_input(x, ctx_patient)
    t1 = time.perf_counter()
    res = he_module.run_he_inference(enc, ctx_server)
    t2 = time.perf_counter()
    prob = HEInference.decrypt_result(res, ctx_patient)
    t3 = time.perf_counter()
    t_enc.append(t1-t0); t_srv.append(t2-t1); t_dec.append(t3-t2)

    show = i < 3 or i == N_INFER - 1
    if i == 3 and N_INFER > 4:
        print(f"  {'...':>4}")
    if show:
        tot = (t3-t0)*1e3
        print(f"  {i:>4}  {t_enc[-1]*1e3:>9.1f}ms  {t_srv[-1]*1e3:>9.1f}ms"
              f"  {t_dec[-1]*1e3:>9.1f}ms  {tot:>9.1f}ms  {prob:.3f}")

t_total = [e+s+d for e,s,d in zip(t_enc, t_srv, t_dec)]
section("Aggregate statistics")
for label, arr, unit, scale in [
    ("Encrypt (patient side)",  t_enc, "ms", 1e3),
    ("Server compute (HE)",     t_srv, "ms", 1e3),
    ("Decrypt (patient side)",  t_dec, "ms", 1e3),
    ("TOTAL per request",       t_total, "ms", 1e3),
]:
    a = np.array(arr)*scale
    print(f"  {label:<30}  avg={a.mean():7.1f} {unit}  std={a.std():5.1f}  "
          f"min={a.min():6.1f}  max={a.max():6.1f}")


# -----------------------------------------------------------------------------
# 2. Plaintext Inference
# -----------------------------------------------------------------------------
banner("2 / 3  Plaintext Inference  (no HE)")

with torch.no_grad():
    _ = model(torch.tensor(X_plain[0]).unsqueeze(0))   # warmup

t_plain = []
with torch.no_grad():
    for x in X_plain:
        t0 = time.perf_counter()
        torch.sigmoid(model(torch.tensor(x).unsqueeze(0))).item()
        t_plain.append(time.perf_counter() - t0)

a = np.array(t_plain)*1e6
section(f"Aggregate statistics  (n={N_PLAIN})")
print(f"  {'Plaintext inference':<30}  avg={a.mean():7.1f} us  std={a.std():5.1f}  "
      f"min={a.min():6.1f}  max={a.max():6.1f}")

overhead = np.mean(t_total) / np.mean(t_plain)
ratio    = 1 / overhead
section("HE vs Plaintext comparison")
print(f"  Plaintext  {bar(1.0)}  {np.mean(t_plain)*1e6:.0f} us")
print(f"  HE         {bar(1.0)}  {np.mean(t_total)*1e3:.0f} ms  ({overhead:.0f}x slower)")
print(f"\n  Privacy cost : +{np.mean(t_total)*1e3:.0f} ms per request")
print(f"  Plaintext is {overhead:.0f}x faster - HE adds {np.mean(t_total)*1e3:.0f} ms to protect patient features")


# -----------------------------------------------------------------------------
# 3. SS+DH vs Plain Concat
# -----------------------------------------------------------------------------
banner("3 / 3  Plain / SS / SS+DH comparison  (training batch)")

print("""
  +-----------------------------------------------------------------+
  |  Plain concat  (no privacy)                                     |
  |    torch.cat([emb_A, emb_B, emb_C])                            |
  |    Coordinator sees all raw embeddings.                         |
  |                                                                 |
  |  SS only  (additive secret sharing, no transmission mask)       |
  |    emb_i --additive_split--> [share_i0, share_i1, share_i2]    |
  |    Shares distributed as-is — eavesdropper on the wire         |
  |    can intercept a share, but cannot reconstruct emb_i alone.   |
  |                                                                 |
  |  SS + DH masking  (current full implementation)                 |
  |    Same as SS, but each share_ij is masked with r_ij before     |
  |    sending; r_ij + r_ji = 0 so masks cancel on aggregation.    |
  |    Even a wiretapper who captures a masked share sees only      |
  |    random noise — cannot link it to any embedding.              |
  +-----------------------------------------------------------------+
""")

dh_masker = DHMasker(NUM_CLIENTS)

def make_embs(bs):
    return [torch.randn(bs, EMB_DIM, requires_grad=True) for _ in range(NUM_CLIENTS)]

# Plain concat
t_plain_cat = []
for _ in range(N_BATCHES):
    embs = make_embs(BATCH_SIZE)
    t0 = time.perf_counter()
    torch.cat(embs, dim=1)
    t_plain_cat.append(time.perf_counter() - t0)

# SS only (no DH masking)
t_ss_only = []
for _ in range(N_BATCHES):
    embs = make_embs(BATCH_SIZE)
    t0 = time.perf_counter()
    shares = [additive_split(e, n=NUM_CLIENTS) for e in embs]
    cat_shares = []
    for j in range(NUM_CLIENTS):
        received = [shares[i][j] for i in range(NUM_CLIENTS)]
        cat_shares.append(torch.cat(received, dim=1))
    t_ss_only.append(time.perf_counter() - t0)

# SS + DH masking
t_ss_dh = []
for _ in range(N_BATCHES):
    embs = make_embs(BATCH_SIZE)
    t0 = time.perf_counter()
    dh_masker.refresh((BATCH_SIZE, EMB_DIM))
    shares = [additive_split(e, n=NUM_CLIENTS) for e in embs]
    cat_shares = []
    for j in range(NUM_CLIENTS):
        received = [dh_masker.recv(dh_masker.send(shares[i][j], i, j), j, i)
                    for i in range(NUM_CLIENTS)]
        cat_shares.append(torch.cat(received, dim=1))
    t_ss_dh.append(time.perf_counter() - t0)

cat_us    = np.array(t_plain_cat) * 1e6
ss_us     = np.array(t_ss_only)   * 1e6
ss_dh_us  = np.array(t_ss_dh)     * 1e6

section(f"Aggregate statistics  (n={N_BATCHES} batches x {BATCH_SIZE} samples)")
print(f"  {'Plain concat':<30}  avg={cat_us.mean():7.1f} us  std={cat_us.std():5.1f}")
print(f"  {'SS only':<30}  avg={ss_us.mean():7.1f} us  std={ss_us.std():5.1f}")
print(f"  {'SS + DH masking':<30}  avg={ss_dh_us.mean():7.1f} us  std={ss_dh_us.std():5.1f}")

section("Visual comparison (per batch)")
max_us = ss_dh_us.mean()
print(f"  Plain   {bar(cat_us.mean()/max_us)}  {cat_us.mean():.1f} us")
print(f"  SS      {bar(ss_us.mean()/max_us)}  {ss_us.mean():.1f} us  "
      f"(+{ss_us.mean()-cat_us.mean():.1f} us vs plain,  {ss_us.mean()/cat_us.mean():.0f}x)")
print(f"  SS+DH   {bar(1.0)}  {ss_dh_us.mean():.1f} us  "
      f"(+{ss_dh_us.mean()-ss_us.mean():.1f} us vs SS,   {ss_dh_us.mean()/ss_us.mean():.0f}x)")

section("DH masking incremental cost")
dh_cost = ss_dh_us.mean() - ss_us.mean()
print(f"  SS only -> SS+DH : +{dh_cost:.1f} us per batch  ({dh_cost/BATCH_SIZE:.2f} us per sample)")
print(f"  DH adds pairwise mask gen + apply/remove ({NUM_CLIENTS*(NUM_CLIENTS-1)//2} pairs)")
print(f"  => Wire-level security for +{dh_cost:.1f} us extra per batch")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
banner("SUMMARY")
print(f"""
  Galaxy Watch Inference (per request)
  +-----------------------------------------------------------+
  |  Stage              |  Without HE  |  With HE (CKKS)     |
  +-----------------------------------------------------------+
  |  Patient encrypt    |    0         |  {np.mean(t_enc)*1e3:>7.1f} ms         |
  |  Server compute     |  {np.mean(t_plain)*1e6:>5.0f} us     |  {np.mean(t_srv)*1e3:>7.1f} ms         |
  |  Patient decrypt    |    0         |  {np.mean(t_dec)*1e3:>7.1f} ms         |
  |  TOTAL              |  {np.mean(t_plain)*1e6:>5.0f} us     |  {np.mean(t_total)*1e3:>7.1f} ms         |
  |  Privacy            |  NONE        |  Patient features   |
  |                     |              |  hidden from server  |
  +-----------------------------------------------------------+
  HE privacy overhead   : {overhead:.0f}x  (plain {np.mean(t_plain)*1e6:.0f} us  ->  HE {np.mean(t_total)*1e3:.0f} ms)

  Training overhead (per batch, batch_size={BATCH_SIZE})
  +-----------------------------------------------------------+
  |  Plain concat               |  {cat_us.mean():>6.1f} us                 |
  |  SS only                    |  {ss_us.mean():>6.1f} us  (+{ss_us.mean()-cat_us.mean():.1f} us vs plain)  |
  |  SS + DH masking            |  {ss_dh_us.mean():>6.1f} us  (+{ss_dh_us.mean()-ss_us.mean():.1f} us vs SS)    |
  |  Privacy gain               |  Embeddings never exposed  |
  +-----------------------------------------------------------+
""")
