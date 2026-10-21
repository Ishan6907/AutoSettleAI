import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import time
from autosettle_vision import pipeline as P
from autosettle_vision.vggt import VGGT3DChecker

IN_DIR = Path(r"c:\Hackathon Confluentia\Aequor\test_photos\indian_accidents")
OUT_DIR = IN_DIR / "annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

test_imgs = sorted([p for p in IN_DIR.glob("in_acc_*.jpg") if p.is_file()])[:5]
print(f"Testing pipeline on first {len(test_imgs)} images...")

vggt = VGGT3DChecker()

COLORS = {
    "Dent": (0, 140, 255),          # Amber / Orange
    "Broken part": (0, 0, 230),     # Red
    "Missing part": (200, 0, 200),  # Purple
    "Surface damage": (0, 200, 200) # Cyan
}

for p in test_imgs:
    t0 = time.time()
    img = cv2.imread(str(p))
    if img is None:
        print(f"Skipping unreadable {p.name}")
        continue
    
    analysis = P.analyse_image(img, p.name)
    parts = analysis.get("parts", [])
    findings = analysis.get("findings", [])
    elapsed = (time.time() - t0) * 1000

    annotated = img.copy()
    
    # 1. Draw parts in translucent / thin white
    for prt in parts:
        poly = prt.get("polygon")
        if poly is not None and len(poly) > 2:
            cv2.polylines(annotated, [np.asarray(poly, np.int32)], True, (255, 255, 255), 1)

    # 2. Draw damages
    for f in findings:
        c = COLORS.get(f.get("damage_class"), (0, 255, 0))
        box = f.get("damage_box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 2)
            
            # VGGT 3D verification
            v_res = vggt.verify_dent(img, box, damage_conf=f.get("damage_conf", 0.5))
            vggt_tag = " [3D-Dent]" if v_res.is_physical_dent else " [2D-Flat]"

            label = f"{f['part_class']}: {f['damage_class']} ({f.get('severity','?')}){vggt_tag}"
            
            # Draw filled background for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = OUT_DIR / f"annotated_{p.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Processed {p.name} ({img.shape[1]}x{img.shape[0]}) in {elapsed:.1f}ms: {len(parts)} parts, {len(findings)} damages -> {out_path.name}")
