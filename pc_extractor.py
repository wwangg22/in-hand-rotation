import torch
import trimesh
import functools
from rl_games.algos_torch.pointnets import PointNet
ckpt_path = "/home/william/Downloads/last_z-axis-working-objsem-w-rot-32dim-new_ep_16000_rew_220.98907.pth"

obj_path = "assets/urdf/objects/meshes/custom/lipton_tea/textured.obj"

ckpt      = torch.load(ckpt_path, map_location="cpu")

# 1️⃣  isolate only the PointNet tensors
prefix   = "a2c_network.pc_encoder."
pc_state = {k[len(prefix):]: v for k, v in ckpt["model"].items()
            if k.startswith(prefix)}
print("PointNet state keys:", pc_state.keys()   )

# 2️⃣  build a matching PointNet (same point_channel/output_dim)
pc_encoder = PointNet(point_channel=3, output_dim=32)

# 3️⃣  load the weights
missing, unexpected = pc_encoder.load_state_dict(pc_state, strict=True)
print("missing:", missing, "unexpected:", unexpected)  # both should be []


N_SAMPLES = 256          # repeats per mesh
PC_SIZE   = 100          # points fed to PointNet (match train‑time)
OUT_DIM   = 32           # embedding size (match your checkpoint)

@functools.lru_cache(maxsize=None)
def load_vertices(fname: str) -> torch.Tensor:
    """Load mesh vertices, cached to avoid disk rereads."""
    verts = trimesh.load(fname, force='mesh').vertices          # (V,3) numpy
    return torch.as_tensor(verts, dtype=torch.float32)          # to torch

def pointnet_embed(verts: torch.Tensor,
                   pc_net: PointNet,
                   pc_size: int = PC_SIZE) -> torch.Tensor:
    """Single forward pass with a random subsample."""
    if len(verts) >= pc_size:
        idx = torch.randperm(len(verts))[:pc_size]
    else:                           # sample‑with‑replacement if mesh is tiny
        idx = torch.randint(0, len(verts), (pc_size,))
    pc = verts[idx].unsqueeze(0)    # (1, N, 3)
    with torch.no_grad():
        feat, _ = pc_net(pc)
    return feat.squeeze(0).cpu()    # (OUT_DIM,)

verts = load_vertices(obj_path)     # (V,3)  CPU tensor
embeds = torch.stack([pointnet_embed(verts, pc_encoder) for _ in range(N_SAMPLES)])
# embeds: (N_SAMPLES, OUT_DIM)

# per‑dimension standard deviation (σ)
per_dim_std = embeds.std(dim=0, unbiased=False)  # (OUT_DIM,)

print(f"σ per dim  (first 10): {per_dim_std[:10].numpy()}")
print(f"σ average: {per_dim_std.mean().item():.5f}")