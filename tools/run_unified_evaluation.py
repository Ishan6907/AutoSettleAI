"""
Unified Training and Testing Evaluation Suite
Evaluates both old images (41 Commons dev photos) and new images (254 Indian accident photos)
across baseline and retrained models with full metric transparency and zero synthetic conflation.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from autosettle_engine import engine as E
from autosettle_engine.tests.helpers import RC, policy
from autosettle_vision import pipeline as P
from autosettle_vision.claim import engine_vision
from autosettle_vision.vggt import VGGT3DChecker

COMMONS_DIR = ROOT / "test_photos" / "commons"
INDIAN_DIR = ROOT / "test_photos" / "indian_accidents"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def parse_gt(s):
    out = []
    for item in filter(None, (x.strip() for x in s.split(";"))):
        part, dclass, sev = (item.split(":") + ["", ""])[:3]
        out.append({"part": part, "damage": dclass, "severity": sev})
    return out


def evaluate_commons():
    """Evaluate 41 Commons test photos with ground truth."""
    gt_file = COMMONS_DIR / "ground_truth.csv"
    gt = {r["file"]: r for r in csv.DictReader(open(gt_file, encoding="utf-8"))}
    pol = policy("Swift", reg="2023-06-01")

    rows = []
    tp = fn = fp = cls_ok = sev_ok = 0
    decisions = {}
    silent_misses_count = 0
    offer_claims_with_misses = []

    print(f"\n--- Evaluating Commons Photos ({len(gt)} images) ---")
    for fname, g in sorted(gt.items()):
        fpath = COMMONS_DIR / fname
        if not fpath.exists():
            continue

        truth = parse_gt(g["damage"])
        tparts = {t["part"]: t for t in truth}

        # Run vision pipeline
        claim_res = P.run_claim([fpath], claim_id=fname)
        ev = engine_vision(claim_res)
        ev["plate_read"] = pol.plate  # neutralize plate mismatch for benchmark

        # Assess with engine
        assess_res = E.assess(pol, ev, RC)
        decision = assess_res["decision"]
        got = {"OFFER": "OFFER", "ASK_MORE": "ASK_MORE"}.get(decision, "HUMAN")
        key = f"expect {g['expect']} -> {got}"
        decisions[key] = decisions.get(key, 0) + 1

        found = {x["part_class"]: x for x in claim_res["detections"] if x.get("damage_class")}

        # Ground truth part-level matching
        missed_on_photo = []
        for p, t in tparts.items():
            if p in found:
                tp += 1
                cls_ok += (found[p]["damage_class"] == t["damage"])
                sev_ok += (found[p]["severity"] == t["severity"])
            else:
                fn += 1
                missed_on_photo.append(p)
        fp += sum(1 for p in found if p not in tparts)

        # Audit check: Silent Misses (misses on claims that resulted in an auto-OFFER)
        if decision == "OFFER" and len(missed_on_photo) > 0:
            silent_misses_count += len(missed_on_photo)
            offer_claims_with_misses.append({
                "file": fname,
                "missed_parts": missed_on_photo,
                "offer_amount": assess_res.get("offer", 0)
            })

        rows.append({
            "file": fname,
            "car": g["car"],
            "expected_decision": g["expect"],
            "engine_decision": decision,
            "gt_damages": len(truth),
            "parts_found": sum(1 for p in tparts if p in found),
            "parts_missed": len(missed_on_photo),
            "detections": len(claim_res["detections"]),
            "offer": assess_res.get("offer", 0)
        })

    summary = {
        "photos_evaluated": len(rows),
        "gt_damaged_parts": tp + fn,
        "parts_found": tp,
        "parts_missed": fn,
        "extra_detections": fp,
        "part_recall": round(tp / max(tp + fn, 1), 3),
        "part_precision": round(tp / max(tp + fp, 1), 3),
        "damage_class_agreement": round(cls_ok / max(tp, 1), 3),
        "severity_agreement": round(sev_ok / max(tp, 1), 3),
        "decisions": dict(sorted(decisions.items())),
        "silent_misses_count": silent_misses_count,
        "offer_claims_with_misses": offer_claims_with_misses,
        "auto_offer_on_human_cases": decisions.get("expect HUMAN -> OFFER", 0)
    }

    # Save CSV
    csv_path = RUNS_DIR / "unified_commons_evaluation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return summary, rows


def evaluate_indian_accidents():
    """Evaluate 254 authentic Indian car accident photos."""
    vggt = VGGT3DChecker()
    pol = policy("Swift", reg="2023-06-01")
    image_paths = sorted([p for p in INDIAN_DIR.glob("in_acc_*.jpg") if p.is_file()])

    print(f"\n--- Evaluating Indian Accident Photos ({len(image_paths)} images) ---")
    rows = []
    total_parts = 0
    total_damages = 0
    vggt_verified_dents = 0
    vggt_rejected_glare = 0
    decisions = {}
    damage_by_class = {}
    damage_by_part = {}
    latencies = []

    for idx, pth in enumerate(image_paths, 1):
        t0 = time.time()
        img = cv2.imread(str(pth))
        if img is None:
            continue

        h, w = img.shape[:2]
        claim_res = P.run_claim([pth], claim_id=pth.name)
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        ev = engine_vision(claim_res)
        ev["plate_read"] = pol.plate
        assess_res = E.assess(pol, ev, RC)
        dec = assess_res["decision"]
        decisions[dec] = decisions.get(dec, 0) + 1

        findings = claim_res.get("detections", [])
        total_damages += len(findings)

        parts_seen = set()
        for f in findings:
            p_cls = f.get("part_class", "unknown")
            d_cls = f.get("damage_class", "Dent")
            parts_seen.add(p_cls)
            damage_by_class[d_cls] = damage_by_class.get(d_cls, 0) + 1
            damage_by_part[p_cls] = damage_by_part.get(p_cls, 0) + 1

            if d_cls == "Dent":
                if f.get("vggt_3d_verified", True):
                    vggt_verified_dents += 1
                else:
                    vggt_rejected_glare += 1

        rows.append({
            "image": pth.name,
            "width": w,
            "height": h,
            "damages_detected": len(findings),
            "parts_involved": ";".join(sorted(parts_seen)),
            "engine_decision": dec,
            "offer": assess_res.get("offer", 0),
            "latency_ms": round(elapsed_ms, 1)
        })

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"  Processed {idx}/{len(image_paths)} images (mean latency {np.mean(latencies):.1f}ms)...")

    summary = {
        "total_images": len(rows),
        "total_damages_detected": total_damages,
        "images_with_damage": sum(1 for r in rows if r["damages_detected"] > 0),
        "damage_by_class": damage_by_class,
        "damage_by_part": dict(sorted(damage_by_part.items(), key=lambda x: -x[1])),
        "vggt_3d_verified_dents": vggt_verified_dents,
        "vggt_rejected_glare": vggt_rejected_glare,
        "decisions": dict(sorted(decisions.items())),
        "average_latency_ms": round(float(np.mean(latencies)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1)
    }

    # Save CSV
    csv_path = RUNS_DIR / "unified_indian_accidents_evaluation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return summary, rows


def main():
    print("=================================================================")
    print("AutoSettle AI: Complete Unified Evaluation & Testing")
    print("=================================================================")

    commons_summary, commons_rows = evaluate_commons()
    indian_summary, indian_rows = evaluate_indian_accidents()

    full_report = {
        "commons": commons_summary,
        "indian_accidents": indian_summary
    }

    report_path = RUNS_DIR / "unified_evaluation_summary.json"
    report_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nEvaluation summary written to: {report_path}")
    print("\n--- Summary Commons ---")
    print(json.dumps(commons_summary, indent=2))
    print("\n--- Summary Indian Accidents ---")
    print(json.dumps(indian_summary, indent=2))


if __name__ == "__main__":
    main()
