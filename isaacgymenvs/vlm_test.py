import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

import torch
from PIL import Image
import open_clip

from isaacgymenvs.tasks.reward_model import CLIPEmbed, CLIPReward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True,
                        help="path to RGB image file")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="projection strength (0 = vanilla CLIP, 1 = 1-D)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. backbone + preprocess -------------------------------------------------
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.to(device).eval()

    embedder = CLIPEmbed(clip_model, transform=preprocess).to(device)

    # 2. reward module ---------------------------------------------------------
    target_prompt   = ["a person kneeling"]          # ★ change to your goal
    baseline_prompt = ["a person standing"]          # ★ change to your baseline
    reward_fn = CLIPReward(
        embed            = embedder,
        target_prompts   = target_prompt,
        baseline_prompts = baseline_prompt,
        alpha            = args.alpha,
    ).to(device).eval()

    # 3. load image ------------------------------------------------------------
    img = Image.open(args.image).convert("RGB")
    img_tensor = torch.tensor(img).unsqueeze(0)       # (1,H,W,3) uint8
    # 4. compute reward --------------------------------------------------------
    with torch.no_grad():
        emb  = embedder(img_tensor.to(device))
        r    = reward_fn(emb)

    print(f"Reward for {args.image}: {r.item():.4f}")

if __name__ == "__main__":
    main()