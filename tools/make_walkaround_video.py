"""
Generates a realistic vehicle walkaround video from real damaged vehicle photography.
Simulates a phone camera walkaround:
- Continuous trajectory panning across panels
- Hand-held camera tremor / micro-motion
- Natural lighting and exposure variations
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMMONS = ROOT / "test_photos" / "commons"
OUT_VIDEO = ROOT / "test_photos" / "walkaround_car_damage_demo.mp4"


def generate_walkaround(
    src_image_path: Path = COMMONS / "c02_Chloe_s_car_after_encounter_with_her_let.jpg",
    out_video_path: Path = OUT_VIDEO,
    fps: int = 30,
    duration_s: float = 3.0,
    out_size: tuple = (1280, 720),
):
    img = cv2.imread(str(src_image_path))
    if img is None:
        raise FileNotFoundError(f"Source image not found: {src_image_path}")

    h, w = img.shape[:2]
    total_frames = int(fps * duration_s)

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, out_size)

    # Walkaround camera trajectory: smooth panning from right to left with subtle zoom & handheld shake
    np.random.seed(42)
    crop_w, crop_h = int(w * 0.75), int(h * 0.75)

    for i in range(total_frames):
        t = i / float(total_frames)
        # Smooth ease-in ease-out trajectory
        ease = 0.5 - 0.5 * np.cos(np.pi * t)

        # Pan across vehicle
        start_x = int(w * 0.05)
        end_x = int(w * 0.20)
        curr_x = int(start_x + ease * (end_x - start_x))

        start_y = int(h * 0.10)
        end_y = int(h * 0.18)
        curr_y = int(start_y + ease * (end_y - start_y))

        # Add realistic handheld micro-jitter
        jitter_x = int(np.random.normal(0, 1.5))
        jitter_y = int(np.random.normal(0, 1.2))

        x1 = max(0, min(w - crop_w, curr_x + jitter_x))
        y1 = max(0, min(h - crop_h, curr_y + jitter_y))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        crop = img[y1:y2, x1:x2]
        frame = cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)

        # Subtle natural lighting fluctuation
        bright_factor = 1.0 + 0.02 * np.sin(2 * np.pi * t * 2)
        frame = np.clip(frame.astype(np.float32) * bright_factor, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    print(f"[Video Generator] Walkaround car video created: {out_video_path} ({total_frames} frames, {duration_s}s)")


if __name__ == "__main__":
    generate_walkaround()
