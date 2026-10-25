#!/usr/bin/env python3
"""V5.1: plaintext validation threshold selection and held-out CKKS evaluation."""
import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 20261024
TARGET_SENSITIVITY = 0.85


def read_csv(name):
    with (ROOT / name).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    out = {r['nsrrid']: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f'Duplicate nsrrid in {name}')
    return out


def model_tokens(name):
    lines = (ROOT / name).read_text().splitlines()
    result = {}
    for line in lines[1:]:
        fields = line.split()
        if fields:
            result[fields[0]] = fields[1:]
    return result


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def counts(rows, threshold, field):
    tp = tn = fp = fn = 0
    for row in rows:
        pred = int(row[field] >= threshold)
        y = row['label']
        if y == 1 and pred == 1: tp += 1
        elif y == 0 and pred == 0: tn += 1
        elif y == 0: fp += 1
        else: fn += 1
    return tp, tn, fp, fn


def metrics(rows, threshold, field):
    tp, tn, fp, fn = counts(rows, threshold, field)
    n = len(rows)
    return dict(n=n, threshold=threshold, TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=(tp + tn) / n,
                sensitivity=tp / (tp + fn) if tp + fn else float('nan'),
                specificity=tn / (tn + fp) if tn + fp else float('nan'),
                precision=tp / (tp + fp) if tp + fp else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                roc_auc=auc(rows, field))


def auc(rows, field):
    pos = sum(r['label'] for r in rows)
    neg = len(rows) - pos
    if not pos or not neg:
        return float('nan')
    ordered = sorted(rows, key=lambda r: r[field])
    rank = 1
    pos_rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][field] == ordered[i][field]:
            j += 1
        average_rank = (i + 1 + j) / 2
        pos_rank_sum += sum(r['label'] for r in ordered[i:j]) * average_rank
        i = j
    return (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def write_csv(name, rows, fields):
    with (ROOT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    a, b, c = (read_csv(n) for n in ('A.csv', 'B.csv', 'C.csv'))
    preds = read_csv('v5_predictions.csv')
    ids = set(a)
    if not (ids == set(b) == set(c) == set(preds)):
        raise ValueError('Patient IDs do not match across CSV files')
    client = model_tokens('client_model.txt')
    features = client['FEATURE_NAMES']
    means = list(map(float, client['SCALER_MEAN']))
    scales = list(map(float, client['SCALER_SCALE']))
    bias = float(client['BIAS'][0])
    weights = [None] * len(features)
    for h in range(3):
        hospital = model_tokens(f'hospital{h}_model.txt')
        for idx, weight in zip(hospital['INDICES'], hospital['WEIGHTS']):
            idx = int(idx)
            if weights[idx] is not None:
                raise ValueError('Duplicate weight index')
            weights[idx] = float(weight)
    if any(w is None for w in weights):
        raise ValueError('Incomplete model weights')

    all_rows = []
    for pid in sorted(ids):
        source = {**a[pid], **b[pid], **c[pid]}
        x = [float(source[name]) for name in features]
        logit = bias + sum(w * (v - mu) / scale for w, v, mu, scale in zip(weights, x, means, scales))
        ahi = float(source['ahi_a0h3a'])
        ckks = float(preds[pid]['ckks_probability'])
        y = int(ahi >= 15)
        if y != int(preds[pid]['true_label']):
            raise ValueError(f'AHI label mismatch for {pid}')
        all_rows.append(dict(nsrrid=pid, ahi=ahi, label=y,
                             plaintext_logit=logit, plaintext_probability=sigmoid(logit),
                             ckks_probability=ckks))

    # Stratified, deterministic split. Validation receives 30% from each class.
    rng = random.Random(SEED)
    validation, test = [], []
    for label in (0, 1):
        subset = [r for r in all_rows if r['label'] == label]
        rng.shuffle(subset)
        n_val = round(len(subset) * 0.30)
        validation.extend(subset[:n_val])
        test.extend(subset[n_val:])

    # All unique plaintext probabilities are sufficient candidate cutoffs.
    candidates = sorted({r['plaintext_probability'] for r in validation})
    candidates = sorted(set([0.0, 0.4, 1.0] + candidates))
    evaluated = [metrics(validation, t, 'plaintext_probability') for t in candidates]
    eligible = [m for m in evaluated if m['sensitivity'] >= TARGET_SENSITIVITY]
    if not eligible:
        raise ValueError('No threshold satisfies target sensitivity')
    best = max(eligible, key=lambda m: (m['accuracy'], m['specificity'], m['threshold']))
    threshold = best['threshold']
    results = [
        dict(dataset='validation_plaintext_optimal', **best),
        dict(dataset='test_plaintext_optimal', **metrics(test, threshold, 'plaintext_probability')),
        dict(dataset='test_ckks_optimal', **metrics(test, threshold, 'ckks_probability')),
        dict(dataset='test_ckks_baseline_0.4', **metrics(test, 0.4, 'ckks_probability')),
    ]
    fields = ['dataset', 'n', 'threshold', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'sensitivity',
              'specificity', 'precision', 'f1', 'roc_auc']
    write_csv('v5_1_metrics.csv', results, fields)
    write_csv('v5_1_threshold_scan.csv', evaluated,
              ['n', 'threshold', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'sensitivity',
               'specificity', 'precision', 'f1', 'roc_auc'])
    out = []
    for row in test:
        r = dict(row)
        r['predicted_label'] = int(row['ckks_probability'] >= threshold)
        r['probability_abs_error'] = abs(row['ckks_probability'] - row['plaintext_probability'])
        out.append(r)
    write_csv('v5_1_test_predictions.csv', out,
              ['nsrrid', 'ahi', 'label', 'plaintext_logit', 'plaintext_probability',
               'ckks_probability', 'predicted_label', 'probability_abs_error'])
    (ROOT / 'v5_1_optimal_threshold.txt').write_text(f'{threshold:.17g}\n')
    print(f'Validation={len(validation)}  Test={len(test)}  Seed={SEED}')
    print(f'Optimal plaintext threshold={threshold:.8f}  minimum validation sensitivity={TARGET_SENSITIVITY:.0%}')
    for result in results:
        print(f"{result['dataset']}: accuracy={result['accuracy']:.4%} "
              f"sensitivity={result['sensitivity']:.4%} specificity={result['specificity']:.4%} "
              f"f1={result['f1']:.4f} AUC={result['roc_auc']:.4f}")
    mae = sum(r['probability_abs_error'] for r in out) / len(out)
    print(f'Test CKKS-vs-plaintext probability MAE={mae:.9g}')
    print('Saved: v5_1_metrics.csv, v5_1_threshold_scan.csv, '
          'v5_1_test_predictions.csv, v5_1_optimal_threshold.txt')


if __name__ == '__main__':
    main()
