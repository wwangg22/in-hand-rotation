from typing import List
import open_clip, torch
import torch.nn as nn

# ----------  tiny helper: use open_clip's own transform  ----------
def default_image_transform(model_name="ViT-B-32"):
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    return preprocess        # torchvision.Compose

# ----------  CLIPEmbed  ----------
class CLIPEmbed(nn.Module):
    def __init__(self, clip_model, transform=None):
        super().__init__()
        self.clip_model = clip_model
        self.transform  = transform or default_image_transform()
    @torch.inference_mode()
    def forward(self, imgs: torch.Tensor):
        if imgs.dtype == torch.uint8:        # HWC uint8 → CHW float
            imgs = imgs.float() / 255.0
        if imgs.shape[1] != 3:               # assume NHWC
            imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.transform(imgs)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            return self.clip_model.encode_image(imgs, normalize=True)

# ----------  CLIPReward  ----------
class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        embed: CLIPEmbed,
        target_prompts: List[str],
        baseline_prompts: List[str],
        alpha: float = 0.5,
    ):
        super().__init__()
        self.embed = embed
        # --- encode prompts ---
        tk = open_clip.tokenize
        text_enc = embed.clip_model.encode_text
        with torch.no_grad():
            g = text_enc(tk(target_prompts)).float().mean(0, keepdim=True)
            b = text_enc(tk(baseline_prompts)).float().mean(0, keepdim=True)
        d = g - b
        # --- register as buffers so .cuda(), .eval() propagate ---
        for name, tensor in dict(target=g, baseline=b, direction=d).items():
            self.register_buffer(name, tensor / tensor.norm())
        # --- projection matrix (Eq. 3 in the paper) ---
        self.alpha = alpha
        self.register_buffer("projection", self._make_P(alpha))
    def _make_P(self, alpha):
        d = self.direction
        P = d.T @ d / (d.norm() ** 2)                # outer product / |d|²
        I = torch.eye(P.size(0), device=P.device)
        return alpha * P + (1 - alpha) * I
    def update_alpha(self, alpha: float):
        self.alpha = alpha
        self.projection.copy_(self._make_P(alpha))
    @torch.inference_mode()
    def forward(self, img_embeddings: torch.Tensor):
        x = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
        diff = (x - self.target) @ self.projection
        return 1 - 0.5 * (diff.norm(dim=-1) ** 2)
