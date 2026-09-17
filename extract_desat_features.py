"""Parse NSRR SHHS1 event-annotation XML files (annotations-events-nsrr/shhs1)
into per-subject SpO2-desaturation summary features.

Deliberately uses ONLY "SpO2 desaturation" events (SaO2-derived, scored from
the oximetry channel alone) rather than respiratory events (obstructive/
central apnea, hypopnea). Respiratory-event counts are what ahi_a0h3a is
computed from, so using them as a training feature would just reconstruct
the label (leakage) and wouldn't match anything a watch could ever sense
(no airflow/effort belt). Desaturation events are still SpO2-only, i.e. the
same modality the watch already provides -- this is "more granular signal
from a sensor we already have", not "a new sensor we don't have".

Output per subject (indexed by nsrrid):
    n_desat            -- count of SpO2 desaturation events in the recording
    odi                -- n_desat / recording_hours  (Oxygen Desaturation Index)
    mean_desat_duration -- mean event duration (seconds)
    mean_desat_drop     -- mean (SpO2Baseline - SpO2Nadir) (%)
"""

import re
import sys
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

FNAME_RE = re.compile(r"shhs1-(\d+)-nsrr\.xml$")


def parse_one(xml_path: str):
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    recording_seconds = None
    n_desat = 0
    durations = []
    drops = []

    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "")
        if concept.startswith("Recording Start Time"):
            dur = ev.findtext("Duration")
            recording_seconds = float(dur) if dur else None
        elif concept.startswith("SpO2 desaturation"):
            n_desat += 1
            dur = ev.findtext("Duration")
            if dur:
                durations.append(float(dur))
            nadir = ev.findtext("SpO2Nadir")
            baseline = ev.findtext("SpO2Baseline")
            if nadir and baseline:
                drops.append(float(baseline) - float(nadir))

    if recording_seconds is None or recording_seconds <= 0:
        return None

    hours = recording_seconds / 3600.0
    return {
        "n_desat": n_desat,
        "odi": n_desat / hours,
        "mean_desat_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_desat_drop": float(np.mean(drops)) if drops else 0.0,
    }


def extract_all(xml_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(f"{xml_dir}/*.xml")):
        m = FNAME_RE.search(path)
        if not m:
            continue
        nsrrid = int(m.group(1))
        feats = parse_one(path)
        if feats is None:
            continue
        feats["nsrrid"] = nsrrid
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df.set_index("nsrrid") if len(df) else df


if __name__ == "__main__":
    xml_dir = sys.argv[1] if len(sys.argv) > 1 else "shhs/polysomnography/annotations-events-nsrr/shhs1"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "shhs/datasets/desat_features.csv"

    df = extract_all(xml_dir)
    df.to_csv(out_csv)
    print(f"[extract_desat_features] {len(df)}명 파싱 완료 -> {out_csv}")
    if len(df):
        print(df.describe())
