#!/usr/bin/env python3
"""
vlm_test.py – simple sanity-check: compute CLIP-based reward for one image.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import open_clip

from reward_model import CLIPEmbed, CLIPReward


# ------------------------------------------------------------------
def load_image_uint8(path: Path) -> torch.Tensor:
    """Return (1, H, W, 3) uint8 tensor suitable for CLIPEmbed."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).unsqueeze(0)  # add batch dim


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="RGB image file")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Projection strength (0 = raw CLIP, 1 = 1-D)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) CLIP backbone + preprocess
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )
    (clip_model
        .eval()                    # inference mode
        .requires_grad_(False)     # no gradients
        .to(device)
    )
    embedder = CLIPEmbed(clip_model, transform=preprocess).to(device)

    # 2) Reward function
    reward_fn = CLIPReward(
        embed=embedder,
        target_prompts = [
            # GOOD  ➜ cylinder orthogonal to the palm, plenty of free handle
            "robot hand presenting a metal cylinder sideways, shaft sticking out clearly",
            "cylinder held perpendicular to fingers, long handle exposed for human grasp",
            "robot rotates tube 90° away from palm, free end offered for pickup",
            "bar protrudes to the side of robotic palm, ready for easy hand-over",
            "robotic gripper shows cylinder horizontally, ample clearance around handle"
        ],

        baseline_prompts = [
            # BAD  ➜ cylinder aligned with palm / hidden / cramped
            "robot hand clamping a cylinder lengthwise, bar flush against palm",
            "cylinder gripped in line with robotic fingers, little handle showing",
            "cylinder falling out of robot's hand",
            "tube buried inside robot's palm, no room for human grasping",
            "robot squeezes metal rod straight along hand, handle obstructed",
            "bar grasped centrally, cylinder mostly covered by robotic fingers"
        ],
        alpha=args.alpha,
    ).to(device).eval().requires_grad_(False)

    # 3) Load image
    img_tensor = load_image_uint8(args.image).to(device)

    # 4) Compute reward
    with torch.no_grad():
        emb = embedder(img_tensor)
        reward = reward_fn(emb)

    print(f"Reward for {args.image}: {reward.item():.4f}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
