import torch
from rl_games.algos_torch.pointnets import PointNet
ckpt_path = "/home/william/Downloads/last_z-axis-working-objsem-w-rot_ep_22000_rew_465.5295.pth"

ckpt      = torch.load(ckpt_path, map_location="cpu")

# 1️⃣  isolate only the PointNet tensors
prefix   = "a2c_network.pc_encoder."
pc_state = {k[len(prefix):]: v for k, v in ckpt["model"].items()
            if k.startswith(prefix)}
print("PointNet state keys:", pc_state.keys()   )

# 2️⃣  build a matching PointNet (same point_channel/output_dim)
pc_encoder = PointNet(point_channel=3)

# 3️⃣  load the weights
missing, unexpected = pc_encoder.load_state_dict(pc_state, strict=True)
print("missing:", missing, "unexpected:", unexpected)  # both should be []

# 4️⃣  quick sanity-check
with torch.no_grad():
    dummy_pts = torch.randn(2, 100, 3)   # (B, N, 3)
    feat, _   = pc_encoder(dummy_pts)
print("output:", feat.shape)