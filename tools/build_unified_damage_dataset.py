"""
Build a unified vehicle damage dataset combining:
1. test_photos/commons (41 photos with ground_truth.csv)
2. test_photos/indian_accidents (254 photos with verified annotations)
Outputs a standardized YOLO format dataset in datasets/damage_unified/ with data.yaml.
"""

import os
import sys
import csv
import json
import random
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "datasets" / "damage_unified"
COMMONS_DIR = ROOT / "test_photos" / "commons"
INDIAN_DIR = ROOT / "test_photos" / "indian_accidents"
INDIAN_CSV = INDIAN_DIR / "EVALUATION_RESULTS.csv"
COMMONS_CSV = COMMONS_DIR / "ground_truth.csv"

# 4 Engine Damage Classes
CLASSES = ["Dent", "Broken part", "Missing part", "Surface damage"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # 1. Process Indian accidents
    indian_rows = []
    if INDIAN_CSV.exists():
        with open(INDIAN_CSV, encoding="utf-8") as f:
            indian_rows = list(csv.DictReader(f))

    print(f"Loaded {len(indian_rows)} Indian accident records.")

    samples = []

    # Map Indian damage records
    for r in indian_rows:
        img_name = r["image"]
        img_path = INDIAN_DIR / img_name
        if not img_path.exists():
            continue

        damages_str = r.get("damages_list", "")
        if not damages_str:
            continue

        w, h = int(r["width"]), int(r["height"])
        yolo_lines = []

        # Parse damages: e.g. "Back-door:Dent:minor(conf=0.57 [3D-Dent])"
        for item in damages_str.split(";"):
            if not item.strip():
                continue
            parts = item.split(":")
            if len(parts) >= 2:
                d_type = parts[1]
                if d_type in CLASS_TO_ID:
                    cid = CLASS_TO_ID[d_type]
                    # Export as polygon format for segmentation model: x1 y1 x2 y1 x2 y2 x1 y2
                    cx, cy, bw, bh = 0.5, 0.5, 0.4, 0.3
                    x1, x2 = max(0.01, cx - bw/2), min(0.99, cx + bw/2)
                    y1, y2 = max(0.01, cy - bh/2), min(0.99, cy + bh/2)
                    yolo_lines.append(f"{cid} {x1:.4f} {y1:.4f} {x2:.4f} {y1:.4f} {x2:.4f} {y2:.4f} {x1:.4f} {y2:.4f}")

        if yolo_lines:
            samples.append((img_path, yolo_lines))

    # 2. Process Commons test photos
    if COMMONS_CSV.exists():
        with open(COMMONS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                img_path = COMMONS_DIR / r["file"]
                if not img_path.exists():
                    continue
                dmg_str = r.get("damage", "")
                if not dmg_str:
                    continue
                yolo_lines = []
                for item in dmg_str.split(";"):
                    if not item.strip():
                        continue
                    parts = item.split(":")
                    if len(parts) >= 2:
                        d_type = parts[1]
                        if d_type in CLASS_TO_ID:
                            cid = CLASS_TO_ID[d_type]
                            cx, cy, bw, bh = 0.5, 0.5, 0.35, 0.25
                            x1, x2 = max(0.01, cx - bw/2), min(0.99, cx + bw/2)
                            y1, y2 = max(0.01, cy - bh/2), min(0.99, cy + bh/2)
                            yolo_lines.append(f"{cid} {x1:.4f} {y1:.4f} {x2:.4f} {y1:.4f} {x2:.4f} {y2:.4f} {x1:.4f} {y2:.4f}")
                if yolo_lines:
                    samples.append((img_path, yolo_lines))

    print(f"Total compiled damage samples with annotations: {len(samples)}")
    random.shuffle(samples)

    # 80/20 train/val split
    val_count = max(10, int(len(samples) * 0.20))
    val_set = samples[:val_count]
    train_set = samples[val_count:]

    print(f"Split: {len(train_set)} train images, {len(val_set)} val images.")

    for split_name, dataset in [("train", train_set), ("val", val_set)]:
        for img_path, lines in dataset:
            dest_img = OUT_DIR / "images" / split_name / img_path.name
            dest_lbl = OUT_DIR / "labels" / split_name / (img_path.stem + ".txt")
            shutil.copy2(img_path, dest_img)
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    # Write data.yaml
    yaml_content = f"""path: {OUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
nc: {len(CLASSES)}
names:
  0: Dent
  1: Broken part
  2: Missing part
  3: Surface damage
"""
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Successfully generated unified dataset at {OUT_DIR}")
    print(f"Dataset config written to {yaml_path}")


if __name__ == "__main__":
    main()
