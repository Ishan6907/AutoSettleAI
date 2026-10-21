"""
Threshold sweep to optimize Precision without hurting Recall.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from autosettle_vision import pipeline as P
from autosettle_vision.claim import engine_vision
from autosettle_engine import engine as E
from autosettle_engine.tests.helpers import RC, policy
import csv

COMMONS_DIR = Path("test_photos/commons")
gt = {r["file"]: r for r in csv.DictReader(open(COMMONS_DIR / "ground_truth.csv", encoding="utf-8"))}

def parse_gt(s):
    out = []
    for item in filter(None, (x.strip() for x in s.split(";"))):
        part, dclass, sev = (item.split(":") + ["", ""])[:3]
        out.append({"part": part, "damage": dclass, "severity": sev})
    return out

# Existing files
existing = {f: g for f, g in gt.items() if (COMMONS_DIR / f).exists()}
print(f"Testing on {len(existing)} existing Commons photos...")

for conf_thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    # Temporarily set CONF_DET
    old_conf = P.CONF_DET
    P.CONF_DET = conf_thresh
    
    tp = fn = fp = 0
    for f, g in sorted(existing.items()):
        truth = parse_gt(g["damage"])
        tparts = {t["part"]: t for t in truth}
        v = P.run_claim([COMMONS_DIR / f], claim_id=f)
        found = {x["part_class"]: x for x in v["detections"] if x.get("damage_class")}
        
        for p in tparts:
            if p in found:
                tp += 1
            else:
                fn += 1
        fp += sum(1 for p in found if p not in tparts)
    
    P.CONF_DET = old_conf
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    print(f"Conf={conf_thresh:.2f} -> TP={tp:2d}, FP={fp:2d}, FN={fn:2d} | Precision={prec*100:5.1f}%, Recall={rec*100:5.1f}%, F1={f1:.3f}")
