"""
Auto-Settle AI vision pipeline: photos / video in, vision JSON out (the contract autosettle_engine.assess() reads).

    python -m autosettle_vision.pipeline IMG [IMG ...] [--video V] [--plate-ocr] [--out claim.json]

Multi-stage perception architecture:
1. Stage 03A: YOLO Part Segmentation (WHERE): outlines vehicle panels, bumpers, glass, and lights.
2. Stage 03B: Topological Disambiguation: corrects spatial class assignments (front/rear wheels, lights).
3. Stage 03C: Part-Guided ROI Cropping & Multi-Scale Damage Detection (WHAT):
   Captures fine dents, scratches, and broken parts that downsampling misses.
4. Stage 04: ByteTrack Multi-Frame Tracking & Temporal Deduplication for Video:
   Tracks parts and damage instances across frames, performs GMC motion compensation, and selects canonical views.
5. Stage 04B: VGGT 3D Grounding & Specular Reflection Filter:
   Verifies physical depth concavity (nabla^2 Z) and temporal UV stability to reject paint reflections and glare.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .tracker import VideoByteTracker
from .vggt import VGGT3DChecker

ROOT = Path(__file__).resolve().parent.parent
W_PARTS = ROOT / "backup" / "parts_s_best.pt"
W_DET = ROOT / "backup" / "dent_det_best.pt"
W_MERGED = ROOT / "backup" / "damage_merged_s_best.pt"

# Alternative weight locations in repo
W_WEIGHTS_PARTS = ROOT / "weights" / "parts_yolo11_best.pt"
W_WEIGHTS_CARDD = ROOT / "weights" / "cardd_yolov11_seg_best.pt"

# detector class -> (compatible part classes, engine damage class). First part is the default if no mask matches.
DET_MAP = {
    "Front-windscreen-damage": (("Windshield",), "Broken part"),
    "Headlight-damage": (("Headlight",), "Broken part"),
    "Rear-windscreen-Damage": (("Back-windshield",), "Broken part"),
    "Runningboard-Damage": (("Rocker-panel",), "Dent"),
    "Sidemirror-Damage": (("Mirror",), "Broken part"),
    "Taillight-Damage": (("Tail-light",), "Broken part"),
    "bonnet-dent": (("Hood",), "Dent"),
    "boot-dent": (("Trunk",), "Dent"),
    "doorouter-dent": (("Front-door", "Back-door"), "Dent"),
    "fender-dent": (("Fender",), "Dent"),
    "front-bumper-dent": (("Front-bumper",), "Dent"),
    "quaterpanel-dent": (("Quarter-panel",), "Dent"),
    "rear-bumper-dent": (("Back-bumper",), "Dent"),
    "roof-dent": (("Roof",), "Dent"),
}

# CarDD classes -> (compatible parts, engine damage class)
CARDD_MAP = {
    "crack": ("Broken part", ("Windshield", "Back-windshield", "Front-bumper", "Back-bumper", "Headlight", "Tail-light")),
    "dent": ("Dent", ("Front-door", "Back-door", "Hood", "Trunk", "Fender", "Quarter-panel", "Front-bumper", "Back-bumper", "Roof", "Rocker-panel")),
    "glass shatter": ("Broken part", ("Windshield", "Back-windshield", "Front-window", "Back-window")),
    "lamp broken": ("Broken part", ("Headlight", "Tail-light")),
    "scratch": ("Surface damage", ("Front-door", "Back-door", "Hood", "Trunk", "Fender", "Quarter-panel", "Front-bumper", "Back-bumper")),
}

# Generic parts model mapping (e.g. BUMPER -> Front-bumper / Back-bumper)
GENERIC_PART_MAP = {
    "BUMPER": ("Front-bumper", "Back-bumper"),
    "DOOR": ("Front-door", "Back-door"),
    "HOOD": ("Hood",),
    "BOOT": ("Trunk",),
    "FENDER": ("Fender", "Quarter-panel"),
    "HEADLIGHT": ("Headlight",),
    "TAILLIGHT": ("Tail-light",),
    "WINDSHIELD": ("Windshield", "Back-windshield"),
    "MIRROR": ("Mirror",),
}

# parts the detector already covers; the weak merged model is not trusted there
DET_PARTS = {p for parts, _ in DET_MAP.values() for p in parts}

IMGSZ = 640
CONF_PARTS = 0.25
CONF_DET = 0.38             # Precision-calibrated threshold to filter false reflections & wheel noise
CONF_MERGED = 0.40          # High-confidence segmentation threshold
MERGED_CLASSES = {"Missing part", "Broken part", "Dent", "Surface damage"}

SEV_THRESHOLDS = (0.30, 0.60)
HEAVY_SEVERE_PARTS = 2      # 2+ severe parts, or severe hood plus one more = heavy crash
BOX_FILL = 0.785            # Area ratio of elliptical dent in box (~pi/4)
MIN_OVERLAP = 0.20          # Share of damage region that must fall on part mask
MIN_OVERLAP_NAMED = 0.05    # Lower bar when damage class already names panel
BLUR_MIN = 60.0
DARK_MIN = 40.0
EDIT_SOFTWARE = ("photoshop", "gimp", "snapseed", "picsart", "lightroom", "canva", "facetune", "pixlr")

_models = {}
_vggt_checker: Optional[VGGT3DChecker] = None


def vggt_checker() -> VGGT3DChecker:
    global _vggt_checker
    if _vggt_checker is None:
        _vggt_checker = VGGT3DChecker()
    return _vggt_checker


def _resolve_weight(primary: Path, fallbacks: List[Path]) -> str:
    if primary.exists():
        return str(primary)
    for fb in fallbacks:
        if fb.exists():
            return str(fb)
    return str(ROOT / "yolo11s-seg.pt")


def models(parts_weight=None, det_weight=None, merged_weight=None, reload=False):
    """Lazily load parts model, primary damage detector, and auxiliary damage segmenter."""
    global _models
    if reload:
        _models.clear()
    if not _models:
        import os
        from ultralytics import YOLO

        p_env = os.environ.get("AUTOSETTLE_PARTS_WEIGHT")
        d_env = os.environ.get("AUTOSETTLE_DET_WEIGHT")
        m_env = os.environ.get("AUTOSETTLE_MERGED_WEIGHT")

        parts_path = parts_weight or p_env or _resolve_weight(W_PARTS, [ROOT / "weights" / "carparts_unified_best.pt", W_WEIGHTS_PARTS, ROOT / "weights" / "parts_s_best.pt"])
        det_path = det_weight or d_env or _resolve_weight(W_DET, [W_WEIGHTS_CARDD, ROOT / "weights" / "dent_det_best.pt"])
        merged_path = merged_weight or m_env or _resolve_weight(W_MERGED, [ROOT / "weights" / "unified_damage_best.pt", W_WEIGHTS_CARDD, ROOT / "weights" / "damage_merged_s_best.pt"])

        _models.update(
            parts=YOLO(str(parts_path)),
            det=YOLO(str(det_path)),
            merged=YOLO(str(merged_path)),
        )
    return _models


def severity_of(frac):
    if frac is None:
        return "minor"
    lo, hi = SEV_THRESHOLDS
    return "minor" if frac < lo else ("moderate" if frac < hi else "severe")


# ------------------------------------------------------------------ quality and integrity
def quality(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    bright = float(grey.mean())
    flags = []
    if blur < BLUR_MIN:
        flags.append("blurry")
    if bright < DARK_MIN:
        flags.append("too_dark")
    return {"blur": round(blur, 1), "brightness": round(bright, 1), "flags": flags}


def dhash(img, size=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def exif_flags(path):
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
    except Exception:
        return [], ["unreadable_metadata"]
    software = str(ex.get(305, "")).lower()
    hard = ["edited_with_" + s for s in EDIT_SOFTWARE if s in software]
    soft = [] if (ex.get(271) or ex.get(272)) else ["no_camera_metadata"]
    return hard, soft


class SeenStore:
    def __init__(self, path=ROOT / "runs" / "seen_hashes.json"):
        self.path = Path(path)
        self.items = json.loads(self.path.read_text()) if self.path.exists() else []

    def match(self, h, claim_id, max_dist=4):
        return [it for it in self.items if it["claim"] != claim_id and hamming(it["h"], h) <= max_dist]

    def add(self, h, claim_id, name):
        self.items.append({"h": h, "claim": claim_id, "name": name})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items))


def normalize_video_path(v):
    if not v:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("video", {}).get("path") or v.get("path") or v.get("name") or str(v)
    if hasattr(v, "name"):
        return v.name
    return str(v)


def video_frames(path, every_s=0.5, max_frames=50):
    """
    Extract keyframes from video at calibrated interval (default 0.5s) and verify continuity via SIFT/ORB homography.
    """
    path = normalize_video_path(path)
    if not path or not Path(path).exists():
        return [], []
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps * every_s)))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(fr)
        i += 1
    cap.release()

    sift = cv2.SIFT_create(800)
    breaks = []
    prev = None
    for k, fr in enumerate(frames):
        h_fr, w_fr = fr.shape[:2]
        g = cv2.cvtColor(cv2.resize(fr, (640, int(640 * h_fr / w_fr))), cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(g, None)
        if prev is not None and des is not None and prev[1] is not None and len(kp) > 10:
            matcher = cv2.BFMatcher()
            m = matcher.knnMatch(prev[1], des, k=2)
            good = [a for a, *b in m if b and a.distance < 0.75 * b[0].distance]
            inl = 0
            if len(good) >= 8:
                src = np.float32([prev[0][a.queryIdx].pt for a in good])
                dst = np.float32([kp[a.trainIdx].pt for a in good])
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                inl = int(mask.sum()) if mask is not None else 0
            if inl < 12:
                breaks.append(k)
        prev = (kp, des)
    return frames, breaks


# ------------------------------------------------------------------ mask utilities
def _poly_mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    return m


def _box_mask(box, shape):
    m = np.zeros(shape[:2], np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
    return m


# ------------------------------------------------------------------ Topological Disambiguation
def _topological_disambiguation(parts: List[Dict[str, Any]], shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
    """
    Stage 03B: Resolves common topological confusions in car parts:
    - Disambiguates Front-wheel vs Back-wheel using front/rear anchors (Hood/Front-bumper vs Trunk/Back-bumper).
    - Disambiguates Headlight vs Tail-light.
    - Disambiguates Front-door vs Back-door using relative longitudinal position.
    """
    if len(parts) <= 1:
        return parts

    # Compute centroids for each part
    for p in parts:
        if "cx" not in p:
            ys, xs = np.nonzero(p["mask"])
            p["cx"] = float(xs.mean()) if len(xs) > 0 else shape[1] / 2.0
            p["cy"] = float(ys.mean()) if len(ys) > 0 else shape[0] / 2.0

    # Determine vehicle front vs rear anchor in current image
    front_anchors = [p["cx"] for p in parts if p["part_class"] in ("Hood", "Front-bumper", "Headlight", "HOOD", "BUMPER")]
    rear_anchors = [p["cx"] for p in parts if p["part_class"] in ("Trunk", "Back-bumper", "Tail-light", "BOOT", "TAILLIGHT")]

    front_x = float(np.mean(front_anchors)) if front_anchors else None
    rear_x = float(np.mean(rear_anchors)) if rear_anchors else None

    for p in parts:
        cls = p["part_class"]
        cls_lower = cls.lower().replace("_", "-")

        # Standardize 33-class, 22-class, and 21-class part names
        if any(k in cls_lower for k in ("front-door", "front-left-door", "front-right-door")):
            p["part_class"] = "Front-door"
        elif any(k in cls_lower for k in ("back-door", "back-left-door", "back-right-door")):
            p["part_class"] = "Back-door"
        elif "front-bumper" in cls_lower:
            p["part_class"] = "Front-bumper"
        elif "back-bumper" in cls_lower:
            p["part_class"] = "Back-bumper"
        elif "front-wheel" in cls_lower:
            p["part_class"] = "Front-wheel"
        elif "back-wheel" in cls_lower:
            p["part_class"] = "Back-wheel"
        elif "wheel" in cls_lower:
            if front_x is not None and rear_x is not None:
                p["part_class"] = "Front-wheel" if abs(p["cx"] - front_x) < abs(p["cx"] - rear_x) else "Back-wheel"
            else:
                p["part_class"] = "Front-wheel"
        elif "fender" in cls_lower:
            p["part_class"] = "Fender"
        elif "quarter-panel" in cls_lower:
            p["part_class"] = "Quarter-panel"
        elif any(k in cls_lower for k in ("headlight", "front-light", "front-left-light", "front-right-light")):
            p["part_class"] = "Headlight"
        elif any(k in cls_lower for k in ("tail-light", "back-light", "back-left-light", "back-right-light")):
            p["part_class"] = "Tail-light"
        elif "mirror" in cls_lower:
            p["part_class"] = "Mirror"
        elif "hood" in cls_lower:
            p["part_class"] = "Hood"
        elif any(k in cls_lower for k in ("trunk", "boot", "tailgate")):
            p["part_class"] = "Trunk"
        elif "front-glass" in cls_lower or "windshield" in cls_lower:
            p["part_class"] = "Back-windshield" if "back" in cls_lower else "Windshield"
        elif "back-glass" in cls_lower:
            p["part_class"] = "Back-windshield"
        elif "rocker-panel" in cls_lower:
            p["part_class"] = "Rocker-panel"
        elif "license-plate" in cls_lower:
            p["part_class"] = "License-plate"
        elif "grille" in cls_lower:
            p["part_class"] = "Grille"
        elif "roof" in cls_lower:
            p["part_class"] = "Roof"
        elif cls == "BUMPER":
            if front_x is not None and rear_x is not None:
                p["part_class"] = "Front-bumper" if abs(p["cx"] - front_x) < abs(p["cx"] - rear_x) else "Back-bumper"
            else:
                p["part_class"] = "Front-bumper"
        elif cls == "DOOR":
            if front_x is not None and rear_x is not None:
                p["part_class"] = "Front-door" if abs(p["cx"] - front_x) < abs(p["cx"] - rear_x) else "Back-door"
            else:
                p["part_class"] = "Front-door"
        elif cls == "BOOT":
            p["part_class"] = "Trunk"
        elif cls == "HOOD":
            p["part_class"] = "Hood"
        elif cls == "HEADLIGHT":
            p["part_class"] = "Headlight"
        elif cls == "TAILLIGHT":
            p["part_class"] = "Tail-light"
        elif cls == "WINDSHIELD":
            p["part_class"] = "Windshield"

        # Disambiguate Wheels
        if cls in ("Front-wheel", "Back-wheel") and front_x is not None and rear_x is not None:
            if abs(p["cx"] - front_x) < abs(p["cx"] - rear_x):
                p["part_class"] = "Front-wheel"
            else:
                p["part_class"] = "Back-wheel"

        # Disambiguate Lights
        if cls in ("Headlight", "Tail-light") and front_x is not None and rear_x is not None:
            if abs(p["cx"] - front_x) < abs(p["cx"] - rear_x):
                p["part_class"] = "Headlight"
            else:
                p["part_class"] = "Tail-light"

    return parts


# ------------------------------------------------------------------ perception on one image
def analyse_image(img, image_id="img"):
    """
    Performs perception on one image:
    1. YOLO Part Segmentation (Stage 03A) -> WHERE.
    2. Topological Disambiguation (Stage 03B).
    3. Part-Guided ROI Cropping & Damage Detection (Stage 03C) -> WHAT.
    4. VGGT 3D Grounding & Specular Reflection Discrimination.
    """
    M = models()
    shape = img.shape
    checker = vggt_checker()

    # 1. Part Segmentation (YOLO11s-seg)
    pr = M["parts"].predict(img, conf=CONF_PARTS, imgsz=IMGSZ, verbose=False)[0]
    parts = []
    if pr.masks is not None:
        for poly, c, k in zip(pr.masks.xy, pr.boxes.conf.tolist(), pr.boxes.cls.tolist()):
            mk = _poly_mask(poly, shape)
            if mk.sum() > 0:
                parts.append({"part_class": pr.names[int(k)], "conf": float(c), "mask": mk, "poly": poly})

    # 2. Topological Disambiguation
    parts = _topological_disambiguation(parts, shape)

    findings = []

    def assign(dmask, allowed, damage_box=None):
        """
        Multi-tier assignment:
        Tier 1: Maximum mask intersection.
        Tier 2: Centroid inside part polygon.
        Tier 3: Minimum distance to compatible part bounding box.
        """
        area = max(1.0, float(dmask.sum()))
        need = MIN_OVERLAP_NAMED if allowed else MIN_OVERLAP
        best = None

        # Tier 1: Overlap
        for p in parts:
            if allowed and p["part_class"] not in allowed:
                continue
            ov = float((dmask & p["mask"]).sum()) / area
            if ov >= need and (best is None or ov > best[0]):
                best = (ov, p)

        if best:
            return best

        # Tier 2: Centroid containment & proximity
        if damage_box is not None:
            cx = (damage_box[0] + damage_box[2]) / 2.0
            cy = (damage_box[1] + damage_box[3]) / 2.0
            min_dist = float("inf")
            closest = None
            for p in parts:
                if allowed and p["part_class"] not in allowed:
                    continue
                p_cx = p.get("cx", 0.0)
                p_cy = p.get("cy", 0.0)
                dist = float(np.hypot(cx - p_cx, cy - p_cy))
                # If damage center is inside bounding box of part with margin
                ys, xs = np.nonzero(p["mask"])
                if len(xs) > 0 and xs.min() - 30 <= cx <= xs.max() + 30 and ys.min() - 30 <= cy <= ys.max() + 30:
                    if dist < min_dist:
                        min_dist = dist
                        closest = p
            if closest:
                return (0.15, closest)

        return None

    # 3. Damage Detection (Roboflow Detector / CarDD Multi-Head)
    dr = M["det"].predict(img, conf=CONF_DET, imgsz=IMGSZ, verbose=False)[0]

    # Collect damage candidate boxes from primary detector
    det_candidates = []
    for i_det, (box, c, k) in enumerate(zip(dr.boxes.xyxy.tolist(), dr.boxes.conf.tolist(), dr.boxes.cls.tolist())):
        name = dr.names[int(k)]
        poly = dr.masks.xy[i_det] if dr.masks is not None and len(dr.masks.xy) > i_det else None
        det_candidates.append((box, float(c), name, poly))

    # Part-Guided ROI Cropping (High-resolution crop inference for large panels)
    for p in parts:
        if p["part_class"] in ("Front-door", "Back-door", "Fender", "Quarter-panel", "Hood", "Front-bumper", "Back-bumper"):
            ys, xs = np.nonzero(p["mask"])
            if len(xs) > 0 and len(xs) * len(ys) > 20000:
                x1, y1 = max(0, int(xs.min() * 0.95)), max(0, int(ys.min() * 0.95))
                x2, y2 = min(shape[1], int(xs.max() * 1.05)), min(shape[0], int(ys.max() * 1.05))
                crop = img[y1:y2, x1:x2]
                if crop.size > 0 and min(crop.shape[:2]) >= 160:
                    crop_dr = M["det"].predict(crop, conf=CONF_DET, imgsz=640, verbose=False)[0]
                    for c_box, c_conf, c_k in zip(crop_dr.boxes.xyxy.tolist(), crop_dr.boxes.conf.tolist(), crop_dr.boxes.cls.tolist()):
                        c_name = crop_dr.names[int(c_k)]
                        # Map box back to global coordinates
                        g_box = [c_box[0] + x1, c_box[1] + y1, c_box[2] + x1, c_box[3] + y1]
                        det_candidates.append((g_box, float(c_conf), c_name, None))

    # Process all damage candidates
    for box, c, name, poly in det_candidates:
        dmask = _poly_mask(poly, shape) if poly is not None else _box_mask(box, shape)

        # Map to damage class and allowed parts
        if name in DET_MAP:
            allowed, dclass = DET_MAP[name]
        elif name in CARDD_MAP:
            dclass, allowed = CARDD_MAP[name]
        else:
            dclass = "Dent"
            allowed = ("Front-door", "Back-door", "Fender", "Front-bumper", "Hood")

        hit = assign(dmask, set(allowed), damage_box=box)

        # 4. VGGT 3D Check & Specular Reflection Discrimination
        v_res = checker.verify_dent(img, box, damage_conf=c) if dclass == "Dent" else None
        if v_res and not v_res.is_physical_dent and c < 0.50:
            # Drop false positive glares
            continue
        effective_conf = v_res.confidence_adjusted if v_res else c

        # Suppress glass specular reflections on windshields below 0.55 confidence
        if any("windshield" in str(al).lower() for al in allowed) and c < 0.55:
            continue

        if hit:
            p = hit[1]
            inter = float((dmask & p["mask"]).sum())
            p_area = max(1.0, float(p["mask"].sum()))
            frac = min(1.0, (inter if poly is not None else BOX_FILL * inter) / p_area)

            findings.append({
                "part_class": p["part_class"],
                "part_conf": round(p["conf"], 3),
                "part_source": "parts_model",
                "damage_class": dclass,
                "damage_conf": round(float(effective_conf), 3),
                "damage_source": "det:" + name,
                "damage_area_fraction": round(frac, 4),
                "severity": severity_of(frac),
                "part_polygon": np.asarray(p["poly"]).round(1).tolist() if "poly" in p and p["poly"] is not None else None,
                "damage_box": [round(v, 1) for v in box],
                "vggt_3d_verified": v_res.is_physical_dent if v_res else True,
            })
        elif len(parts) == 0 and c >= 0.65:
            # Recover named damage ONLY in extreme macro close-ups where no panel boundaries are visible
            assigned_part = allowed[0]
            if len(allowed) > 1 and "Front-door" in allowed:
                # Disambiguate door based on vehicle longitudinal position
                cx = (box[0] + box[2]) / 2.0
                assigned_part = "Front-door" if cx < shape[1] * 0.55 else "Back-door"

            # Compute estimated area fraction based on box size relative to vehicle context
            approx_panel_area = max(1000.0, float(shape[0] * shape[1] * 0.12))
            est_frac = min(0.85, (dmask.sum() * BOX_FILL) / approx_panel_area)

            findings.append({
                "part_class": assigned_part,
                "part_conf": round(float(effective_conf), 3),
                "part_source": "damage_geometry",
                "damage_class": dclass,
                "damage_conf": round(float(effective_conf), 3),
                "damage_source": "det:" + name,
                "damage_area_fraction": round(est_frac, 4),
                "severity": severity_of(est_frac),
                "damage_box": [round(v, 1) for v in box],
                "note": "geometry recovered in macro close-up",
                "vggt_3d_verified": v_res.is_physical_dent if v_res else True,
            })

    # Secondary / Merged Model (auxiliary broken/missing parts)
    mr = M["merged"].predict(img, conf=CONF_MERGED, imgsz=IMGSZ, verbose=False)[0]
    if mr.masks is not None:
        for poly, c, k in zip(mr.masks.xy, mr.boxes.conf.tolist(), mr.boxes.cls.tolist()):
            name = mr.names[int(k)]
            if name not in MERGED_CLASSES:
                continue
            dmask = _poly_mask(poly, shape)
            hit = assign(dmask, None)
            if not hit or hit[1]["part_class"] in DET_PARTS:
                continue
            p = hit[1]
            frac = float((dmask & p["mask"]).sum()) / max(1.0, float(p["mask"].sum()))
            findings.append({
                "part_class": p["part_class"],
                "part_conf": round(p["conf"], 3),
                "part_source": "parts_model",
                "damage_class": name if name in ("Broken part", "Missing part", "Surface damage") else "Dent",
                "damage_conf": round(float(c), 3),
                "damage_source": "damage_merged_s",
                "damage_area_fraction": round(frac, 4),
                "severity": severity_of(frac),
                "part_polygon": np.asarray(p["poly"]).round(1).tolist() if "poly" in p and p["poly"] is not None else None,
                "damage_polygon": np.asarray(poly).round(1).tolist(),
            })

    plate = next((p for p in sorted(parts, key=lambda p: -p["conf"]) if p["part_class"] == "License-plate"), None)
    plate_box = None
    if plate is not None:
        ys, xs = np.nonzero(plate["mask"])
        if len(xs) > 0 and len(ys) > 0:
            plate_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return {
        "image_id": image_id,
        "findings": findings,
        "parts_seen": sorted({p["part_class"] for p in parts}),
        "parts": parts,
        "plate_box": plate_box,
    }


# ------------------------------------------------------------------ claim-level fusion
def _fuse(per_image, min_support):
    """
    Merge findings by (part, damage class).
    Keeps the worst severity seen in at least `min_support` images.
    """
    groups = {}
    for im in per_image:
        for f in im["findings"]:
            if f.get("part_class") in (None, "unassigned"):
                continue
            groups.setdefault((f["part_class"], f["damage_class"]), []).append({**f, "image_id": im["image_id"]})

    out = []
    sev_rank = {"minor": 0, "moderate": 1, "severe": 2}
    for (part, dclass), fs in sorted(groups.items()):
        if len({f["image_id"] for f in fs}) < min_support:
            continue
        best = max(fs, key=lambda f: (sev_rank.get(f["severity"], 0), f["damage_conf"]))
        best = {**best, "support_images": sorted({f["image_id"] for f in fs})}
        out.append(best)
    return out


# ------------------------------------------------------------------ claim run entrypoint
def run_claim(image_paths=(), video=None, claim_id=None, plate_reader=None, seen=None):
    """
    Returns the vision JSON for one claim.
    Supports single/multi-photo workflows and ByteTrack-powered walkaround video workflows.
    """
    claim_id = claim_id or hashlib.sha1("|".join(map(str, image_paths)).encode()).hexdigest()[:10]
    seen = seen if seen is not None else SeenStore()
    checker = vggt_checker()
    integrity, flags, imgs = [], [], []

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            flags.append(f"unreadable:{Path(p).name}")
            continue
        hard, soft = exif_flags(p)
        integrity += [f"{h}:{Path(p).name}" for h in hard]
        flags += [f"{s}:{Path(p).name}" for s in soft]
        imgs.append((Path(p).name, img))

    tracked_entities = []
    min_support = 1

    # ------------------ Video Processing with ByteTrack & VGGT ------------------
    video = normalize_video_path(video)
    if video:
        frames, breaks = video_frames(video)
        if breaks:
            integrity.append(f"video_not_continuous:frames_{breaks[:5]}")
        imgs += [(f"{Path(video).name}#f{i}", fr) for i, fr in enumerate(frames)]

        # Run per-frame perception
        analyzed_frames = []
        for i, fr in enumerate(frames):
            frame_name = f"{Path(video).name}#f{i}"
            analysis = analyse_image(fr, frame_name)
            analyzed_frames.append((frame_name, fr, analysis))

        # Run ByteTrack multi-frame tracking & deduplication
        byte_tracker = VideoByteTracker(min_support_frames=2 if len(frames) >= 4 else 1)
        tracked_entities = byte_tracker.track_video_detections(analyzed_frames, vggt_checker=checker)

        min_support = 2 if len(frames) >= 4 else 1

    # Duplicates check inside claim and against historical claims
    hashes = {}
    for name, img in imgs:
        h = dhash(img)
        if not name.count("#f"):
            for other, oh in hashes.items():
                if hamming(h, oh) <= 2:
                    flags.append(f"duplicate_within_claim:{name}={other}")
            for m in seen.match(h, claim_id):
                integrity.append(f"photo_seen_in_claim_{m['claim']}:{name}")
            hashes[name] = h

    # Multi-view 3D consistency check across static claim photos
    if len(imgs) >= 2 and not video:
        mv_res = checker.check_multiview_consistency([im for _, im in imgs[:6]])
        if not mv_res.is_consistent:
            integrity.append(f"multiview_3d_inconsistent:score_{mv_res.consistency_score}")

    per_image, quality_report = [], {}
    for name, img in imgs:
        q = quality(img)
        quality_report[name] = q
        if q["flags"]:
            flags.append(f"low_quality:{name}:{','.join(q['flags'])}")
            continue
        # Use existing analysis if already computed during video tracking
        if video:
            im_analysis = next((af[2] for af in analyzed_frames if af[0] == name), None)
            if im_analysis:
                per_image.append(im_analysis)
                continue
        per_image.append(analyse_image(img, name))

    # Deduplicate and fuse detections
    if tracked_entities and video:
        # Construct fused detections directly from ByteTrack consolidated entities
        dets = []
        for te in tracked_entities:
            if not te.is_verified_physical:
                continue
            dets.append({
                "part_class": te.part_class,
                "part_conf": te.max_confidence,
                "damage_class": te.damage_class,
                "damage_conf": te.max_confidence,
                "severity": te.fused_severity,
                "damage_area_fraction": te.damage_area_fraction,
                "track_id": te.track_id,
                "canonical_frame": te.canonical_frame_name,
                "support_frames": te.support_frames_count,
                "damage_source": "bytetrack_consolidated",
            })
    else:
        dets = _fuse(per_image, min_support)

    if any(f.get("part_class") == "unassigned" for im in per_image for f in im["findings"]):
        flags.append("unassigned_damage")

    # Heavy crash & structural integrity gate (safety rules -> surveyor)
    severe = {d["part_class"] for d in dets if d["severity"] == "severe"}
    all_damaged_parts = {d["part_class"] for d in dets}
    front_damaged = {d["part_class"] for d in dets if d["part_class"] in ("Hood", "Front-bumper", "Fender", "Headlight", "Grille")}

    is_heavy_crash = (
        len(severe) >= HEAVY_SEVERE_PARTS
        or ("Hood" in severe and len(severe) >= 2)
        or {"Front-door", "Back-door"} <= severe
        or ("Hood" in all_damaged_parts and "Front-bumper" in all_damaged_parts)
        or ("Hood" in all_damaged_parts and "Fender" in all_damaged_parts)
        or any(d["part_class"] == "Headlight" and d["severity"] == "severe" for d in dets)
        or len(front_damaged) >= 3
        or len(all_damaged_parts) >= 4
    )
    if is_heavy_crash:
        flags.append("structural_suspected")

    # License plate recognition
    plate_read = None
    if plate_reader:
        for (name, img), im in zip([x for x in imgs if not quality_report[x[0]]["flags"]], per_image):
            if im.get("plate_box"):
                x1, y1, x2, y2 = im["plate_box"]
                pad = 6
                crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                plate_read = plate_reader(crop)
                if plate_read:
                    break
        if plate_read is None and imgs:
            plate_read = plate_reader(imgs[0][1])

    for name, h in hashes.items():
        seen.add(h, claim_id, name)
    if not dets and per_image:
        flags.append("no_damage_found")

    return {
        "image_id": claim_id,
        "plate_read": plate_read,
        "integrity_flags": integrity,
        "flags": flags,
        "detections": dets,
        "images": [n for n, _ in imgs],
        "frame_images": {n: img for n, img in imgs},
        "quality": quality_report,
        "findings_by_image": {im["image_id"]: im["findings"] for im in per_image},
        "per_image": [
            {k: v for k, v in im.items() if k not in ("findings", "parts")}
            | {"n_findings": len(im["findings"])}
            for im in per_image
        ],
        "tracked_entities": [te.__dict__ for te in tracked_entities] if tracked_entities else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--video")
    ap.add_argument("--claim-id")
    ap.add_argument("--plate-ocr", action="store_true", help="read the plate with Qwen (loads the VLM)")
    ap.add_argument("--out")
    a = ap.parse_args()
    reader = None
    if a.plate_ocr:
        from .qwen import QwenVL
        reader = QwenVL().read_plate
    v = run_claim(a.images, a.video, a.claim_id, reader)
    txt = json.dumps(v, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt)
    print(json.dumps({k: v[k] for k in ("plate_read", "integrity_flags", "flags")}, indent=1))
    for d in v["detections"]:
        print(f"  {d['part_class']:<14} {str(d['damage_class']):<13} {str(d['severity']):<9} area={d.get('damage_area_fraction')}"
              f"  part_conf={d.get('part_conf')} dmg_conf={d.get('damage_conf')}  [{d.get('damage_source')}]")


if __name__ == "__main__":
    main()
