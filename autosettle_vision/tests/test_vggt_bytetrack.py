"""
Unit and integration tests for VGGT 3D Grounding and ByteTrack Video Tracking.
"""
import unittest
from pathlib import Path

import numpy as np

from autosettle_vision.tracker import VideoByteTracker, TrackedDamageEntity
from autosettle_vision.vggt import VGGT3DChecker, Dent3DVerification, MultiViewConsistencyResult
from autosettle_vision import pipeline as P


class TestVGGT3D(unittest.TestCase):
    def setUp(self):
        self.checker = VGGT3DChecker()
        self.test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a shaded simulated panel
        cv2 = __import__("cv2")
        cv2.rectangle(self.test_img, (50, 50), (590, 430), (160, 160, 160), -1)

    def test_depth_estimation(self):
        depth = self.checker.estimate_depth_map(self.test_img)
        self.assertEqual(depth.shape, (480, 640))
        self.assertTrue(0.0 <= depth.min() <= depth.max() <= 1.0)

    def test_dent_concavity_and_verification(self):
        # Physical dent with stable UV trajectory
        box = [150, 150, 250, 250]
        stable_uv = [(0.3, 0.4), (0.31, 0.4), (0.3, 0.41), (0.29, 0.4)]
        res_true = self.checker.verify_dent(self.test_img, box, damage_conf=0.7, uv_trajectory=stable_uv)
        self.assertIsInstance(res_true, Dent3DVerification)
        self.assertTrue(res_true.is_physical_dent)
        self.assertIsNone(res_true.rejection_reason)

    def test_specular_reflection_rejection(self):
        # Moving reflection with high UV variance across frames
        box = [150, 150, 250, 250]
        moving_uv = [(0.10, 0.20), (0.25, 0.35), (0.45, 0.50), (0.65, 0.65)]
        res_glare = self.checker.verify_dent(self.test_img, box, damage_conf=0.35, uv_trajectory=moving_uv)
        self.assertFalse(res_glare.is_physical_dent)
        self.assertIn("moving_specular_reflection", str(res_glare.rejection_reason))

    def test_multiview_consistency(self):
        # Same image slightly shifted should be consistent
        im1 = self.test_img.copy()
        im2 = np.roll(im1, 5, axis=1)
        res = self.checker.check_multiview_consistency([im1, im2])
        self.assertIsInstance(res, MultiViewConsistencyResult)

    def test_point_cloud_generation(self):
        pts, cols = self.checker.generate_point_cloud(self.test_img, downsample=8)
        self.assertEqual(pts.shape[-1], 3)
        self.assertEqual(cols.shape[-1], 3)
        self.assertEqual(len(pts), len(cols))


class TestByteTrackVideo(unittest.TestCase):
    def setUp(self):
        self.tracker = VideoByteTracker(min_support_frames=2)

    def test_tracker_init(self):
        self.assertIsNotNone(self.tracker.damage_tracker)
        self.assertIsNotNone(self.tracker.part_tracker)

    def test_multi_frame_track_deduplication(self):
        # Simulate 5 consecutive frames where the SAME dent is observed on a front door
        frames = []
        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Simulated front door mask
            mask_door = np.zeros((480, 640), dtype=np.uint8)
            mask_door[100:350, 100:400] = 1
            parts = [{"part_class": "Front-door", "conf": 0.88, "mask": mask_door}]
            # Minor motion in damage box
            dx = i * 2
            findings = [{
                "part_class": "Front-door",
                "damage_class": "Dent",
                "damage_conf": 0.85,
                "damage_box": [180 + dx, 200, 240 + dx, 260],
                "damage_area_fraction": 0.08,
                "severity": "minor",
            }]
            frames.append((f"frame_{i}.jpg", img, {"parts": parts, "findings": findings}))

        entities = self.tracker.track_video_detections(frames)
        # Should be consolidated into 1 unique damage entity instead of 5
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].damage_class, "Dent")
        self.assertEqual(entities[0].part_class, "Front-door")
        self.assertEqual(entities[0].support_frames_count, 5)

    def test_transient_noise_dropped(self):
        # Single frame flicker should be dropped when video has multiple frames
        frames = []
        for i in range(4):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            findings = []
            if i == 1:
                # Appears only once
                findings.append({
                    "part_class": "Front-door",
                    "damage_class": "Dent",
                    "damage_conf": 0.35,
                    "damage_box": [180, 200, 240, 260],
                    "damage_area_fraction": 0.05,
                    "severity": "minor",
                })
            frames.append((f"frame_{i}.jpg", img, {"parts": [], "findings": findings}))

        entities = self.tracker.track_video_detections(frames)
        self.assertEqual(len(entities), 0)


class TestTopologicalDisambiguation(unittest.TestCase):
    def test_wheel_disambiguation(self):
        # Front wheel misclassified as back wheel when close to front bumper
        shape = (480, 640, 3)
        parts = [
            {"part_class": "Front-bumper", "mask": np.zeros((480, 640), dtype=np.uint8), "cx": 100.0, "cy": 300.0},
            {"part_class": "Trunk", "mask": np.zeros((480, 640), dtype=np.uint8), "cx": 550.0, "cy": 250.0},
            {"part_class": "Back-wheel", "mask": np.zeros((480, 640), dtype=np.uint8), "cx": 180.0, "cy": 400.0},
        ]
        resolved = P._topological_disambiguation(parts, shape)
        wheel = next(p for p in resolved if "wheel" in p["part_class"])
        # Should be corrected to Front-wheel because cx (180) is closer to front bumper (100) than trunk (550)
        self.assertEqual(wheel["part_class"], "Front-wheel")


if __name__ == "__main__":
    unittest.main()
