#!/usr/bin/env python3
"""
convert_to_notional_mp4.py
──────────────────────────
Re-encode *any* video (WebM, MP4, AVI, …) to an MP4 that embeds and plays
in Notion (H-264/AVC, YUV 4:2:0).  Uses only OpenCV + its FFmpeg backend;
no external ffmpeg binary or x264 license needed.

Example
-------
    python convert_to_notional_mp4.py --input reward_overlay.mp4 \
                                      --output reward_overlay_notional.mp4 \
                                      --fps keep
"""

import argparse, sys, math
from pathlib import Path
import cv2

def get_source_fps(cap: cv2.VideoCapture) -> float:
    """Return a reliable FPS reading; fallback to 30 if unknown."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and not math.isnan(fps) and fps > 0:
        return fps
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if n <= 0:
        return 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, _ = cap.read()
    dur_ms = cap.get(cv2.CAP_PROP_POS_MSEC) if ok else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n / (dur_ms / 1000.0) if dur_ms > 0 else 30.0

def convert(src_path: Path, dst_path: Path, dst_fps: float):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        sys.exit(f"❌  Could not open {src_path}")

    if dst_fps == 0:                          # keep source
        dst_fps = get_source_fps(cap)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H-264 baseline/main → works in browsers
    writer = cv2.VideoWriter(str(dst_path), fourcc, dst_fps, (W, H))
    if not writer.isOpened():
        sys.exit("❌  VideoWriter failed with codec 'avc1'. "
                 "Install opencv-python-headless>=4.7.0 (FFmpeg enabled).")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        idx += 1
        if idx % 200 == 0:
            print(f"\rconverted {idx} frames", end="", flush=True)

    cap.release(); writer.release()
    print(f"\n✅  Saved {dst_path}  |  {idx} frames  |  FPS={dst_fps:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, type=Path, help="source video")
    ap.add_argument("--output", type=Path,
                    help="destination file (default *_notional.mp4)")
    ap.add_argument("--fps", type=str, default="keep",
                    help="'keep' (default) uses source FPS, "
                         "otherwise specify a number, e.g. 12 or 24")
    args = ap.parse_args()

    src = args.input
    dst = args.output or src.with_stem(src.stem + "_notional").with_suffix(".mp4")

    if args.fps == "keep":
        fps_val = 0        # sentinel → copy source FPS
    else:
        try:
            fps_val = float(args.fps)
            if fps_val <= 0: raise ValueError
        except ValueError:
            sys.exit("❌  --fps must be 'keep' or a positive number.")
    convert(src, dst, fps_val)

if __name__ == "__main__":
    main()
