#!/usr/bin/env python3
"""
change_fps.py
─────────────
Re-write a video with a different frame-rate.

Example
-------
    python change_fps.py \
        --input reward_overlay.mp4 \
        --fps   10 \
        --output reward_overlay_10fps.mp4
"""

import argparse
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, type=Path, help="source video")
    ap.add_argument("--output", default=None,   type=Path, help="destination")
    ap.add_argument("--fps",    required=True, type=float, help="new frame-rate")
    args = ap.parse_args()

    src = cv2.VideoCapture(str(args.input))
    if not src.isOpened():
        raise SystemExit(f"Cannot open {args.input}")

    W = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    dst_path = args.output or args.input.with_stem(args.input.stem + f"_{int(args.fps)}fps")
    dst = cv2.VideoWriter(str(dst_path), fourcc, args.fps, (W, H))

    while True:
        ok, frame = src.read()
        if not ok: break
        dst.write(frame)

    src.release()
    dst.release()
    print(f"Saved {dst_path} at {args.fps} fps")

if __name__ == "__main__":
    main()
