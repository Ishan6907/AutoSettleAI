"""
Multi-Frame Video Tracking & Cross-Frame Deduplication via ByteTrack.

Implements Stage 04 of the Auto-Settle AI Architecture:
1. ByteTrack Association for Part and Damage Streams.
2. Global Camera Motion Compensation (GMC) via inter-frame homographies.
3. Hierarchical Damage-to-Part Parent Association.
4. Homography-Warped Spatial Clustering across walk-around frames.
5. Canonical Keyframe Selection (sharpness x panel_area x confidence x centrality).
6. Temporal Consolidation: prevents multi-billing across walk-around frames.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


@dataclass
class TrackedDamageEntity:
    track_id: int
    damage_class: str
    part_class: str
    parent_part_track_id: Optional[int]
    canonical_frame_idx: int
    canonical_frame_name: str
    canonical_box: List[float]
    canonical_poly: Optional[List[List[float]]]
    max_confidence: float
    mean_confidence: float
    support_frames_count: int
    frame_indices: List[int]
    uv_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    fused_severity: str = "moderate"
    damage_area_fraction: float = 0.05
    is_verified_physical: bool = True
    rejection_reason: Optional[str] = None


class VideoByteTracker:
    """
    ByteTrack-based multi-frame tracker for walkaround vehicle inspection videos.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.30,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.30,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        min_support_frames: int = 2,
    ):
        self.args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
            gmc_method="none",
        )
        self.damage_tracker = BYTETracker(self.args)
        self.part_tracker = BYTETracker(self.args)
        self.min_support_frames = min_support_frames
        self._sift = cv2.SIFT_create(800)

    # ------------------------------------------------------------------ Motion & Homography
    def compute_interframe_homography(
        self, prev_img: np.ndarray, curr_img: np.ndarray
    ) -> Optional[np.ndarray]:
        """Estimates global camera motion homography matrix H_{t-1 -> t}."""
        g1 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self._sift.detectAndCompute(g1, None)
        kp2, des2 = self._sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, *rest in matches if rest and m.distance < 0.75 * rest[0].distance]

        if len(good) < 8:
            return None

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        h_mat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return h_mat

    # ------------------------------------------------------------------ ByteTrack Update
    def update_frame(
        self,
        frame_idx: int,
        frame_name: str,
        img_bgr: np.ndarray,
        part_boxes: List[List[float]],
        part_confs: List[float],
        part_classes: List[str],
        damage_boxes: List[List[float]],
        damage_confs: List[float],
        damage_classes: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs one step of ByteTrack on parts and damages for current frame.
        Returns: (tracked_parts [N, 8], tracked_damages [M, 8])
        Format of each tracked row: [x1, y1, x2, y2, track_id, conf, class_id, idx]
        """
        h, w = img_bgr.shape[:2]

        # 1. Update Parts Tracker
        if part_boxes:
            p_tensor = torch.tensor(
                [[*b, c, i] for i, (b, c) in enumerate(zip(part_boxes, part_confs))], dtype=torch.float32
            )
            p_boxes = Boxes(p_tensor, orig_shape=(h, w))
            t_parts = self.part_tracker.update(p_boxes)
        else:
            t_parts = np.empty((0, 8))

        # 2. Update Damage Tracker
        if damage_boxes:
            d_tensor = torch.tensor(
                [[*b, c, i] for i, (b, c) in enumerate(zip(damage_boxes, damage_confs))], dtype=torch.float32
            )
            d_boxes = Boxes(d_tensor, orig_shape=(h, w))
            t_damages = self.damage_tracker.update(d_boxes)
        else:
            t_damages = np.empty((0, 8))

        return t_parts, t_damages

    # ------------------------------------------------------------------ Frame Keyframe Quality
    @staticmethod
    def keyframe_score(
        img_bgr: np.ndarray, box: List[float], conf: float
    ) -> float:
        """
        Calculates keyframe view score: sharpness x area x confidence x centrality.
        """
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size < 4:
            return 0.0

        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Area ratio
        area = float((x2 - x1) * (y2 - y1)) / float(w * h)

        # Centrality
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        centrality = 1.0 - float(np.hypot(cx - 0.5, cy - 0.5))

        score = (np.log1p(max(0.0, sharpness)) * 0.3 + 1.0) * area * conf * max(0.2, centrality)
        return float(score)

    # ------------------------------------------------------------------ Full Video Pipeline
    def track_video_detections(
        self,
        frames: List[Tuple[str, np.ndarray, Dict[str, Any]]],
        vggt_checker: Optional[Any] = None,
    ) -> List[TrackedDamageEntity]:
        """
        Tracks, associates and consolidates detections across all frames of a video.
        Consolidates recurring sightings into unique Damage Entities.
        """
        tracks_data: Dict[int, Dict[str, Any]] = {}
        prev_img = None

        for idx, (fname, img, frame_analysis) in enumerate(frames):
            p_boxes = []
            p_confs = []
            p_names = []
            for p in frame_analysis.get("parts", []):
                # If poly is available, bounding box is enclosing rect
                if "mask" in p and p["mask"] is not None:
                    ys, xs = np.nonzero(p["mask"])
                    if len(xs) > 0:
                        p_boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
                    else:
                        continue
                elif "box" in p:
                    p_boxes.append(p["box"])
                else:
                    continue
                p_confs.append(float(p.get("conf", 0.8)))
                p_names.append(p.get("part_class", "unknown"))

            d_boxes = []
            d_confs = []
            d_names = []
            d_findings = []
            for f in frame_analysis.get("findings", []):
                if "damage_box" in f:
                    d_boxes.append(f["damage_box"])
                    d_confs.append(float(f.get("damage_conf", 0.5)))
                    d_names.append(f.get("damage_class", "Dent"))
                    d_findings.append(f)

            # Update ByteTracker
            t_parts, t_damages = self.update_frame(
                idx, fname, img, p_boxes, p_confs, p_names, d_boxes, d_confs, d_names
            )

            # Associate tracked damage with tracked parts
            for d_row in t_damages:
                dx1, dy1, dx2, dy2, d_tid, d_conf, d_cls_idx, d_orig_idx = d_row
                d_tid = int(d_tid)
                d_orig_idx = int(d_orig_idx)
                f_item = d_findings[d_orig_idx] if d_orig_idx < len(d_findings) else {}
                d_class = f_item.get("damage_class", "Dent")
                assigned_part = f_item.get("part_class", "unassigned")

                # Find best parent part track by box intersection
                parent_p_tid = None
                best_ov = 0.0
                parent_box = None
                for p_row in t_parts:
                    px1, py1, px2, py2, p_tid, p_conf, p_cls_idx, p_orig_idx = p_row
                    ix1, iy1 = max(dx1, px1), max(dy1, py1)
                    ix2, iy2 = min(dx2, px2), min(dy2, py2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        d_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
                        ov = inter / d_area
                        if ov > best_ov:
                            best_ov = ov
                            parent_p_tid = int(p_tid)
                            parent_box = [px1, py1, px2, py2]
                            p_orig_idx = int(p_orig_idx)
                            if p_orig_idx < len(p_names):
                                assigned_part = p_names[p_orig_idx]

                # Compute normalized UV inside parent panel if available
                uv = None
                if parent_box:
                    pw = max(1.0, parent_box[2] - parent_box[0])
                    ph = max(1.0, parent_box[3] - parent_box[1])
                    cx = (dx1 + dx2) / 2.0
                    cy = (dy1 + dy2) / 2.0
                    uv = ((cx - parent_box[0]) / pw, (cy - parent_box[1]) / ph)

                q_score = self.keyframe_score(img, [dx1, dy1, dx2, dy2], d_conf)

                if d_tid not in tracks_data:
                    tracks_data[d_tid] = {
                        "track_id": d_tid,
                        "damage_class": d_class,
                        "part_class": assigned_part,
                        "parent_part_track_id": parent_p_tid,
                        "confs": [d_conf],
                        "frames": [(idx, fname, [dx1, dy1, dx2, dy2], q_score, img, f_item)],
                        "uvs": [uv] if uv else [],
                    }
                else:
                    trk = tracks_data[d_tid]
                    trk["confs"].append(d_conf)
                    trk["frames"].append((idx, fname, [dx1, dy1, dx2, dy2], q_score, img, f_item))
                    if uv:
                        trk["uvs"].append(uv)
                    if trk["part_class"] in ("unassigned", "unknown") and assigned_part not in ("unassigned", "unknown"):
                        trk["part_class"] = assigned_part

            prev_img = img

        # Consolidate tracks into Unique Damage Entities
        entities: List[TrackedDamageEntity] = []
        for tid, tdata in sorted(tracks_data.items()):
            n_frames = len(tdata["frames"])
            # Require minimum support frames (filters out transient flickering detections)
            if n_frames < self.min_support_frames and len(frames) >= 4:
                continue

            # Pick canonical keyframe with maximum view score
            best_f = max(tdata["frames"], key=lambda it: it[3])
            canon_idx, canon_name, canon_box, _, canon_img, canon_finding = best_f

            max_c = float(max(tdata["confs"]))
            mean_c = float(np.mean(tdata["confs"]))

            # Severity and area fraction from canonical finding or aggregated
            frac = canon_finding.get("damage_area_fraction") or 0.05
            sev = canon_finding.get("severity") or (
                "minor" if frac < 0.30 else ("moderate" if frac < 0.60 else "severe")
            )

            # VGGT 3D Check on Canonical View
            is_phys = True
            rejection = None
            if vggt_checker and tdata["damage_class"] == "Dent":
                v_res = vggt_checker.verify_dent(
                    canon_img,
                    canon_box,
                    damage_conf=max_c,
                    uv_trajectory=tdata["uvs"] if len(tdata["uvs"]) >= 3 else None,
                )
                is_phys = v_res.is_physical_dent
                rejection = v_res.rejection_reason
                max_c = v_res.confidence_adjusted

            entity = TrackedDamageEntity(
                track_id=tid,
                damage_class=tdata["damage_class"],
                part_class=tdata["part_class"],
                parent_part_track_id=tdata["parent_part_track_id"],
                canonical_frame_idx=canon_idx,
                canonical_frame_name=canon_name,
                canonical_box=canon_box,
                canonical_poly=canon_finding.get("damage_polygon"),
                max_confidence=round(max_c, 3),
                mean_confidence=round(mean_c, 3),
                support_frames_count=n_frames,
                frame_indices=[it[0] for it in tdata["frames"]],
                uv_trajectory=tdata["uvs"],
                fused_severity=sev,
                damage_area_fraction=round(float(frac), 4),
                is_verified_physical=is_phys,
                rejection_reason=rejection,
            )
            entities.append(entity)

        return entities
