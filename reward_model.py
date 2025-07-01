"""
reward_model.py
Standalone CLIPEmbed + CLIPReward implementation
"""

from typing import List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import open_clip


# ------------------------------------------------------------------
# Helper – default CLIP preprocessing pipeline
# ------------------------------------------------------------------
def default_image_transform(model_name: str = "ViT-B-32"):
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    return preprocess  # torchvision.Compose


# ------------------------------------------------------------------
# CLIPEmbed  –  raw NHWC uint8 tensor  ➜  CLIP image embedding
#               (now converts tensors to PIL internally)
# ------------------------------------------------------------------
class CLIPEmbed(nn.Module):
    def __init__(self, clip_model, transform=None):
        super().__init__()
        self.clip_model = clip_model
        self.transform = transform or default_image_transform()

    @torch.inference_mode()
    def forward(self, imgs: torch.Tensor):
        """
        Accepts tensor of shape (N, H, W, 3) uint8  *or*
        (N, 3, H, W) float/uint8.  Converts each item to PIL,
        applies OpenCLIP's transform, then encodes.
        """
        if not isinstance(imgs, torch.Tensor):
            raise TypeError("CLIPEmbed expects a torch.Tensor input")

        # --- ensure NHWC uint8 for PIL conversion ------------------
        if imgs.dtype != torch.uint8:
            imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)

        if imgs.shape[1] == 3:              # NCHW → NHWC
            imgs = imgs.permute(0, 2, 3, 1)

        # imgs is now (N, H, W, 3) uint8 on *any* device
        imgs_pil = [
            Image.fromarray(img.cpu().numpy()) for img in imgs
        ]

        # apply OpenCLIP preprocessing, stack into batch
        device = next(self.clip_model.parameters()).device
        imgs_proc = torch.stack([self.transform(im) for im in imgs_pil]).to(device)

        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            return self.clip_model.encode_image(imgs_proc, normalize=True)


# ------------------------------------------------------------------
# CLIPReward  –  projection-based reward (paper Sec 3)
# ------------------------------------------------------------------
class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        embed: "CLIPEmbed",
        target_prompts: List[str],
        baseline_prompts: List[str],
        alpha: float = 0.5,
    ):
        super().__init__()
        self.embed = embed

        # encode prompts on same device as the CLIP model
        device = next(embed.clip_model.parameters()).device
        tokenize = open_clip.tokenize
        enc_text = embed.clip_model.encode_text
        with torch.no_grad():
            g = enc_text(tokenize(target_prompts).to(device)).float().mean(0, keepdim=True)
            b = enc_text(tokenize(baseline_prompts).to(device)).float().mean(0, keepdim=True)

        d = g - b
        for name, tensor in dict(target=g, baseline=b, direction=d).items():
            self.register_buffer(name, tensor / tensor.norm())

        self.alpha = alpha
        self.register_buffer("projection", self._make_P(alpha))

    # ---- helpers --------------------------------------------------
    def _make_P(self, alpha: float) -> torch.Tensor:
        d = self.direction
        P = d.T @ d / (d.norm() ** 2)
        I = torch.eye(P.size(0), device=P.device)
        return alpha * P + (1 - alpha) * I

    def update_alpha(self, alpha: float):
        self.alpha = alpha
        self.projection.copy_(self._make_P(alpha))

    # ---- forward --------------------------------------------------
    @torch.inference_mode()
    def forward(self, img_embeddings: torch.Tensor):
        x = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
        diff = (x - self.target) @ self.projection
        return 1.0 - 0.5 * (diff.norm(dim=-1) ** 2)
