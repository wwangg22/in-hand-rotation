#!/usr/bin/env python3
"""
alpha_sweep.py
──────────────
Iterate α ∈ {0.00, 0.05, …, 1.00}, compute the average CLIP-projection
reward gap (standing – sitting) for images in *this* directory, and
save a plot to disk.

Requirements: matplotlib, reward_model.py in the same folder.
"""

from pathlib import Path
from typing import Dict, List
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import open_clip                        # reward_model imports this too

from reward_model import CLIPEmbed, CLIPReward


# ─────────────────────────────────────────────────────────────
def load_image_uint8(p: Path) -> torch.Tensor:
    """(1,H,W,3) uint8 tensor for CLIPEmbed."""
    img = Image.open(p).convert("RGB")
    return torch.tensor(np.array(img, dtype=np.uint8)).unsqueeze(0)


def gather_files(root: Path) -> Dict[str, List[Path]]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    standing = [f for ext in exts for f in root.glob(f"human_standing*{ext}")]
    sitting  = [f for ext in exts for f in root.glob(f"human_sitting*{ext}")]
    return {"standing": sorted(standing), "sitting": sorted(sitting)}


# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_step", type=float, default=0.05,
                        help="Step size for α grid (default 0.05)")
    parser.add_argument("--out", default="alpha_gap.png",
                        help="Output PNG filename")
    args = parser.parse_args()

    alphas = np.arange(0.0, 1.0 + 1e-9, args.alpha_step)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root   = Path.cwd()

    files = gather_files(root)
    if not files["standing"] or not files["sitting"]:
        print("Need images named human_standing* and human_sitting* in:", root)
        return

    # Load ViT-bigG-14 in fp16, inference-only
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )
    clip_model.eval().requires_grad_(False).half().to(device)
    embedder = CLIPEmbed(clip_model, preprocess).to(device)

    gaps = []
    with torch.inference_mode():
        for α in alphas:
            reward_fn = (CLIPReward(
                embed=embedder,
                target_prompts=["a human standing up"],
                    baseline_prompts=["a human"],
                alpha=float(α),
            ).to(device).eval().requires_grad_(False))

            sums, counts = {"standing": 0.0, "sitting": 0.0}, {"standing": 0, "sitting": 0}
            for pose, paths in files.items():
                for p in paths:
                    emb = embedder(load_image_uint8(p).to(device))
                    r   = reward_fn(emb).item()
                    sums[pose]   += r
                    counts[pose] += 1

            gap = (sums["standing"] / counts["standing"]) - (sums["sitting"] / counts["sitting"])
            gaps.append(gap)
            print(f"α={α:4.2f}  gap={gap:6.4f}")

    # ─── plot & save ───
    plt.plot(alphas, gaps, marker="o")
    plt.xlabel("α")
    plt.ylabel("mean reward gap  (standing − sitting)")
    plt.title("Reward separation vs α")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nPlot saved to: {args.out}")


if __name__ == "__main__":
    main()
