"""
Evaluate and label Indian car accident pictures.
Runs the complete post-YOLO segmentation pipeline and VGGT 3D verification,
draws labeled bounding boxes & part polygons, and compiles CSV & Markdown summaries.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from autosettle_vision import pipeline as P
from autosettle_vision.vggt import VGGT3DChecker

IN_DIR = ROOT / "test_photos" / "indian_accidents"
OUT_DIR = IN_DIR / "annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = IN_DIR / "EVALUATION_RESULTS.csv"
MD_OUT = IN_DIR / "EVALUATION_SUMMARY.md"

COLORS = {
    "Dent": (0, 140, 255),           # Amber / Orange
    "Broken part": (0, 0, 230),      # Crimson Red
    "Missing part": (200, 0, 200),   # Purple / Magenta
    "Surface damage": (0, 200, 200)  # Cyan
}

DEFAULT_COLOR = (0, 255, 0)


def draw_label(img, text, x, y, bg_color=(20, 20, 20), text_color=(255, 255, 255), font_scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 3
    y_top = max(0, y - th - pad * 2)
    y_bot = y_top + th + pad * 2
    x_right = min(img.shape[1], x + tw + pad * 2)
    cv2.rectangle(img, (x, y_top), (x_right, y_bot), bg_color, -1)
    cv2.putText(img, text, (x + pad, y_bot - pad - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


def main():
    vggt = VGGT3DChecker()
    image_paths = sorted([p for p in IN_DIR.glob("in_acc_*.jpg") if p.is_file()])
    print(f"Starting evaluation & labeling on {len(image_paths)} Indian accident photos...")

    results = []
    total_parts = 0
    total_damages = 0
    vehicles_with_damage = 0
    damage_by_class = {}
    damage_by_part = {}
    vggt_verified_dents = 0

    t_start = time.time()

    for idx, pth in enumerate(image_paths, 1):
        t0 = time.time()
        img = cv2.imread(str(pth))
        if img is None:
            continue

        h, w = img.shape[:2]
        
        # Run perception pipeline
        try:
            analysis = P.analyse_image(img, pth.name)
        except Exception as e:
            print(f"  [{idx:03d}/{len(image_paths)}] Error analyzing {pth.name}: {e}")
            continue

        parts = analysis.get("parts", [])
        findings = analysis.get("findings", [])
        elapsed = (time.time() - t0) * 1000

        annotated = img.copy()

        # 1. Overlay part contours
        for prt in parts:
            poly = prt.get("polygon")
            pname = prt.get("part_class", "Part")
            if poly is not None and len(poly) > 2:
                pts = np.asarray(poly, np.int32)
                cv2.polylines(annotated, [pts], True, (240, 240, 240), 1, cv2.LINE_AA)

        # 2. Overlay damage boxes & labels
        finding_descriptions = []
        for f in findings:
            p_class = f.get("part_class", "unknown")
            d_class = f.get("damage_class", "Damage")
            sev = f.get("severity", "moderate")
            conf = f.get("damage_conf", 0.5)
            box = f.get("damage_box")

            damage_by_class[d_class] = damage_by_class.get(d_class, 0) + 1
            damage_by_part[p_class] = damage_by_part.get(p_class, 0) + 1

            vggt_tag = ""
            is_3d_dent = False

            if box:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # VGGT 3D concavity check
                if d_class == "Dent":
                    v_res = vggt.verify_dent(img, box, damage_conf=conf)
                    if v_res.is_physical_dent:
                        vggt_tag = " [3D-Dent]"
                        is_3d_dent = True
                        vggt_verified_dents += 1
                    else:
                        vggt_tag = " [2D-Flat]"

                color = COLORS.get(d_class, DEFAULT_COLOR)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                label_text = f"{p_class} | {d_class} ({sev}) {conf:.2f}{vggt_tag}"
                draw_label(annotated, label_text, x1, y1, bg_color=color, text_color=(0, 0, 0) if d_class=="Dent" else (255, 255, 255))
                finding_descriptions.append(f"{p_class}:{d_class}:{sev}(conf={conf:.2f}{vggt_tag})")

        # Save annotated image
        out_name = f"labeled_{pth.name}"
        cv2.imwrite(str(OUT_DIR / out_name), annotated)

        total_parts += len(parts)
        total_damages += len(findings)
        if len(findings) > 0:
            vehicles_with_damage += 1

        rec = {
            "image": pth.name,
            "width": w,
            "height": h,
            "parts_detected": len(parts),
            "damages_detected": len(findings),
            "parts_list": ";".join(sorted(set(prt.get("part_class", "") for prt in parts))),
            "damages_list": ";".join(finding_descriptions),
            "latency_ms": round(elapsed, 1),
            "labeled_path": f"annotated/{out_name}"
        }
        results.append(rec)

        if idx % 20 == 0 or idx == len(image_paths) or len(findings) > 0:
            print(f"  [{idx:03d}/{len(image_paths)}] {pth.name[:35]:<35} -> {len(parts)} parts, {len(findings)} damages ({elapsed:.0f}ms)")

    total_time = time.time() - t_start

    # Write CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "width", "height", "parts_detected", "damages_detected", "parts_list", "damages_list", "latency_ms", "labeled_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Write Markdown Summary Report
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("# Indian Car Accident Dataset: Evaluation & Labeling Report\n\n")
        f.write(f"- **Total Images Evaluated**: {len(results)}\n")
        f.write(f"- **Photos with Confirmed Vehicle Parts**: {sum(1 for r in results if r['parts_detected'] > 0)}\n")
        f.write(f"- **Photos with Detected Structural Damages**: {vehicles_with_damage}\n")
        f.write(f"- **Total Vehicle Parts Segmented**: {total_parts}\n")
        f.write(f"- **Total Damage Instances Localized**: {total_damages}\n")
        f.write(f"- **VGGT 3D Grounding Verified Dents**: {vggt_verified_dents}\n")
        f.write(f"- **Average Inference Latency**: {round(total_time / max(len(results), 1) * 1000, 1)} ms/image\n\n")

        f.write("### Damage Class Breakdown\n")
        f.write("| Damage Class | Instances Detected |\n|---|---|\n")
        for dcls, cnt in sorted(damage_by_class.items(), key=lambda x: -x[1]):
            f.write(f"| {dcls} | {cnt} |\n")

        f.write("\n### Most Damaged Vehicle Parts\n")
        f.write("| Vehicle Part | Damaged Instances |\n|---|---|\n")
        for part, cnt in sorted(damage_by_part.items(), key=lambda x: -x[1])[:10]:
            f.write(f"| {part} | {cnt} |\n")

        f.write("\n### Sample Detected Accidents (Top Highlights)\n")
        f.write("| Image | Dimensions | Parts | Damages Detected | Annotated Output |\n|---|---|---|---|---|\n")
        for r in [r for r in results if r['damages_detected'] > 0][:15]:
            f.write(f"| `{r['image']}` | {r['width']}x{r['height']} | {r['parts_detected']} | {r['damages_list']} | [`{r['labeled_path']}`]({r['labeled_path']}) |\n")

    print("\n" + "=" * 65)
    print(f"EVALUATION & LABELING COMPLETE ({total_time:.1f}s)")
    print(f"Processed: {len(results)} images")
    print(f"Labeled outputs: {OUT_DIR}")
    print(f"CSV Report: {CSV_OUT}")
    print(f"Markdown Summary: {MD_OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
