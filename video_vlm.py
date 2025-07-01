#!/usr/bin/env python3
"""
video_reward_overlay.py
───────────────────────
Read an input video (WebM, MP4, …), compute a CLIP-projection reward per
frame, draw a running reward chart, and save a side-by-side MP4 whose FPS
matches the source.

Example
-------
    python video_reward_overlay.py --input hand_demo.webm --alpha 0.7
"""

import argparse, sys, math
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless backend
import matplotlib.pyplot as plt
import torch, open_clip
from PIL import Image
from reward_model import CLIPEmbed, CLIPReward

# ─────────────────── helpers ────────────────────
def torch_image_from_bgr(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).unsqueeze(0)  # (1,H,W,3) uint8


def init_model(device: str, alpha: float):
    clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )
    clip.eval().requires_grad_(False).half().to(device)
    embedder = CLIPEmbed(clip, preprocess).to(device)
    reward   = (CLIPReward(
        embed=embedder,
        target_prompts=["robot hand presenting a cylinder sideways, handle exposed"],
        baseline_prompts=["robot hand holding a cylinder flush against palm"],
        alpha=alpha,
    ).to(device).eval().requires_grad_(False))
    return embedder, reward


def plot_strip(values, w: int, h: int) -> np.ndarray:
    """Render the reward history as an RGB image (H,W,3) with y-axis 0.4–0.8."""
    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.plot(values, color="tab:blue")
    plt.ylim(0.5, 0.7)                         # ← fixed range
    plt.xlabel("frame")
    plt.ylabel("reward")
    plt.tight_layout()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def get_input_fps(cap: cv2.VideoCapture) -> float:
    """Return source FPS, or derive it if metadata is missing."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and not math.isnan(fps) and fps > 0:
        return fps

    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if n_frames <= 0:
        return 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames - 1)
    ok, _ = cap.read()
    msec = cap.get(cv2.CAP_PROP_POS_MSEC) if ok else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n_frames / (msec / 1000.0) if msec > 0 else 30.0


# ─────────────────── main ───────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, type=Path, help="source video")
    ap.add_argument("--output", default="reward_overlay.mp4", type=Path,
                    help="destination MP4")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="projection strength (0–1)")
    args = ap.parse_args()

    if args.output.suffix.lower() != ".mp4":
        args.output = args.output.with_suffix(".mp4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder, reward_fn = init_model(device, args.alpha)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        sys.exit(f"Could not open input video: {args.input}")

    fps = 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    graph_w = int(H * 1.0)                     # square chart

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(args.output), fourcc, fps, (W + graph_w, H))
    if not out.isOpened():
        sys.exit("Failed to open VideoWriter — check mp4v codec support")

    rewards, idx = [], 0
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            r = reward_fn(embedder(torch_image_from_bgr(frame_bgr).to(device))).item()
            rewards.append(r)

            chart_bgr = cv2.cvtColor(plot_strip(rewards, graph_w, H), cv2.COLOR_RGB2BGR)
            out.write(np.hstack([frame_bgr, chart_bgr]))

            idx += 1
            if idx % 50 == 0:
                print(f"\rprocessed {idx} frames", end="", flush=True)

    cap.release(); out.release()
    print(f"\nSaved side-by-side video to {args.output} (fps={fps:.2f})")


if __name__ == "__main__":
    main()
