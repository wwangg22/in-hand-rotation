
import os, glob, random, torch, collections, gc
import time,argparse, torch.nn as nn, torch.optim as optim
from rl_games.algos_torch.pointnets import PointNet

import trimesh
import functools

from typing import List, Tuple

from rl_games.algos_torch.visual_tactile_transformer import ObjectSemanticsTransformer
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.network_builder import ObjectEncoder

class EpisodeDataset:
    """
    Loads full tensors only when they are actually needed, keeps none.
    Now samples a single random frame per episode, so returned
    tensors have shape (B, D) instead of (B, F, D).
    """

    def __init__(self, folder: str, pattern="*.pt",
                 recursive: bool = False, min_len: int = 1):
        pat = "**/*.pt" if recursive else pattern
        self.paths = sorted(glob.glob(os.path.join(folder, pat),
                                      recursive=recursive))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {folder}")
        print(f"[EpisodeDataset] found {len(self.paths)} chunk files")

        self.episodes: List[Tuple[int, int, int]] = []   # (cid, start, end)
        self.episode_items: List[str] = []               # asset per episode
        self.chunk_order: List[torch.Tensor] = []        # keeps every perm
        self.min_len = max(1, min_len)

        # ---------------- build index -----------------------------------
        for cid, p in enumerate(self.paths):
            done, env_id, assets = self._load_done_env(p)      # raw arrays
            order = torch.argsort(env_id, stable=True)
            self.chunk_order.append(order)

            done_s, env_s = done[order], env_id[order]
            self._index_chunk(cid, done_s, env_s, assets)

        print(f"[EpisodeDataset] indexed {len(self.episodes)} episodes "
              f"(min_len={self.min_len})")

    # -------------------------------------------------------------------
    def _load_done_env(self, path):
        tup = torch.load(path, map_location="cpu")
        return tup[-3], tup[-2], tup[-1]   # done, env_id, assets

    def _index_chunk(self, cid, done, env_id, assets):
        start = 0
        for count in torch.unique_consecutive(env_id, return_counts=True)[1]:
            stop     = start + count
            env_idx  = env_id[start].item()
            item_str = assets[env_idx]

            ep_start = start
            for i in range(start, stop):
                if done[i]:
                    if i - ep_start + 1 >= self.min_len:
                        self.episodes.append((cid, ep_start, i))
                        self.episode_items.append(item_str)
                    ep_start = i + 1
            if stop - ep_start >= self.min_len:
                self.episodes.append((cid, ep_start, stop - 1))
                self.episode_items.append(item_str)
            start = stop

    # -------------------------------------------------------------------
    def __len__(self): 
        return len(self.episodes)

    def _sample_one_episode(self):
        """Pick one episode, then one random frame within it."""
        cid, s, e = self.episodes[random.randrange(len(self.episodes))]
        item_str  = self.episode_items[self.episodes.index((cid, s, e))]
        idx       = random.randint(s, e)           # single frame index
        return cid, idx, item_str

    def _load_chunk(self, cid: int):
        data  = torch.load(self.paths[cid], map_location="cpu")
        order = self.chunk_order[cid]
        # apply same permutation to every tensor we care about
        return dict(
            obs          = data[0][order],
            teacher_obs  = data[1][order],
            actions      = data[2][order],
            sigmas       = data[3][order],
            pointcloud   = data[4][order],
            pc_embedding = data[5][order],
            done         = data[6][order],
            env_id       = data[7][order]
        )

    def sample(self, batch_size: int):
        keys  = ["obs","teacher_obs","actions","sigmas",
                 "pointcloud","done","env_id"]
        batch = {k: [] for k in keys}
        batch["asset"] = []

        choices = [self._sample_one_episode() for _ in range(batch_size)]

        by_chunk = {}
        for cid, idx, item in choices:
            by_chunk.setdefault(cid, []).append((idx, item))

        for cid, idx_item_list in by_chunk.items():
            ch = self._load_chunk(cid)
            for idx, item in idx_item_list:
                for k in keys:
                    batch[k].append(ch[k][idx])
                batch["asset"].append(item)
            del ch
            gc.collect()

        for k in keys:
            batch[k] = torch.stack(batch[k], dim=0)   # (B, D, …)
        return batch
    
DEFAULTS = dict(
    hidden_dim      = 80,      # transformer inner size
    repr_dim        = 80,       # point-cloud embedding size
    sem_dim         = 32,       # pc_embedding target size (= act_dim)
    lr              = 1e-4,
    steps           = 4000,   # optimisation steps, not epochs
    batch_size      = 512,       # episodes per update
    frames_per_ep   = 12,        # timesteps sampled per episode
    log_every       = 25,
)
def _preproc_obs( obs_batch):
    import copy
    if type(obs_batch) is dict:
        obs_batch = copy.copy(obs_batch)
        for k, v in obs_batch.items():
            if v.dtype == torch.uint8:
                obs_batch[k] = v.float() / 255.0
            else:
                obs_batch[k] = v
    else:
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
    return obs_batch

ckpt_path = "/home/william/Downloads/last_z-axis-working-objsem-w-obj-enc-8-dim_ep_6000_rew_1713.696.pth"

@functools.lru_cache(maxsize=None)
def load_vertices(fname: str) -> torch.Tensor:
    """Load mesh vertices, cached to avoid disk rereads."""
    verts = trimesh.load(fname, force='mesh').vertices          # (V,3) numpy
    return torch.as_tensor(verts, dtype=torch.float32)          # to torch

def pointnet_embed(verts: torch.Tensor,
                   pc_net: PointNet,
                   pc_size: int = 100) -> torch.Tensor:
    """Single forward pass with a random subsample."""
    if len(verts) >= pc_size:
        idx = torch.randperm(len(verts))[:pc_size]
    else:                           # sample‑with‑replacement if mesh is tiny
        idx = torch.randint(0, len(verts), (pc_size,))
    pc = verts[idx].unsqueeze(0)    # (1, N, 3)
    with torch.no_grad():
        feat, _ = pc_net(pc)
    return feat.squeeze(0).cpu()    # (OUT_DIM,)

class MLPwPC(nn.Module):
    """
    Simple fully‑connected network.

    Args
    ----
    in_dim : int
        Input feature dimension.
    hidden_dims : list[int]
        Sizes of hidden layers, e.g. [128, 64].
    out_dim : int
        Output dimension.
    activation : torch.nn.Module, optional
        Non‑linearity (default: nn.ReLU).
    dropout : float, optional
        Dropout probability applied after each hidden layer (default: 0).

    Example
    -------
    model = MLP(in_dim=256, hidden_dims=[128, 64], out_dim=10)
    logits = model(torch.randn(32, 256))
    """
    def __init__(self, in_dim, hidden_dims, out_dim,
                 activation=nn.ReLU, dropout=0.0):
        super().__init__()

        layers = []
        last = in_dim + 256 # add 256 for PointNet embedding
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        layers.append(nn.Linear(last, out_dim))
        # layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.pc_encoder = PointNet(point_channel=6)

    def forward(self, x, obj_embed):
        #if x is dict:
        if isinstance(x, dict):
            pc_embed,_ = self.pc_encoder(x['point_cloud'])
            x = torch.cat((x['obs'], pc_embed), dim=-1)
        x = torch.cat((x, obj_embed), dim=-1)  # (B,F,in_dim+256+32)
        return self.net(x) 

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ❶ dataset ----------------------------------------------------------
    ds = EpisodeDataset(
        cfg.data, recursive=True,          # find all .pt under that dir
        min_len=cfg.frames,                # prune short eps
    )

    """

    self.obj_buf[:,  : 7]  = self.object_pose
    self.obj_buf[:,  7:10] = self.object_linvel
    self.obj_buf[:, 10:13] = self.vel_obs_scale * self.object_angvel
    self.obj_buf[:, 13:16] = self.object_semantics          # mass, μ, scale
    """

    # ❷ model ------------------------------------------------------------
    model = MLPwPC(
        in_dim = 340 + 32,
        hidden_dims = [1024, 1024],
        out_dim = 8,              # you hard-coded this
    ).to(device)

    ckpt      = torch.load(ckpt_path, map_location="cpu")

    # trans_model = ObjectSemanticsTransformer(
    #     repr_dim = cfg.repr_dim,
    #     act_dim  = cfg.sem_dim,
    #     hidden_dim = cfg.hidden_dim,
    #     num_feat_per_step = 1,              # you hard-coded this
    #     policy_head = "gmm",      # or "gmm" for GMM output
    # ).to(device)

    # trans_ckpt      = torch.load(trans_ckpt_path, map_location=)
    #     trans_model.load_state_dict(trans_ckpt, strict=True)


    model_ckpt = torch.load("latest_mlp.pt", map_location=device)
    model.load_state_dict(model_ckpt, strict=True)

    

    prefix   = "a2c_network.pc_encoder."
    pc_state = {k[len(prefix):]: v for k, v in ckpt["model"].items()
                if k.startswith(prefix)}
    
    prefix2 = "running_mean_std."
    norm_state = {k[len(prefix2):]: v for k, v in ckpt["model"].items()
                if k.startswith(prefix2)}
    
    pc_encoder = PointNet(point_channel=3, output_dim=32)

    
    missing, unexpected = pc_encoder.load_state_dict(pc_state, strict=True)


    optimiser = optim.AdamW(model.parameters(), lr=cfg.lr)

    teacher_normalization = RunningMeanStd((356,)).to(device)

    teacher_normalization.load_state_dict(norm_state)

    prefix2   = "a2c_network.object_enc."
    pc_state2 = {k[len(prefix2):]: v for k, v in ckpt["model"].items()
                if k.startswith(prefix2)}
    obj_encoder = ObjectEncoder(
        in_dim= 16,
        latent_dim=8
    ).to(device)
    
    obj_encoder.load_state_dict(pc_state2)
    obj_encoder.eval()  # no training on object encoder

    # batch = ds.sample(cfg.batch, cfg.frames)

    # mesh_path = os.path.join(
    #                           "assets/urdf/objects/meshes/custom",
    #                           batch["asset"][1], "textured.obj")

    # verts = load_vertices(mesh_path)     # (V,3)  CPU tensor

    # embeds = pointnet_embed(verts, pc_encoder)

    # ❸ training loop ----------------------------------------------------
    t0 = time.time()
    for step in range(1, cfg.steps + 1):

        # -------- sample batch & send to GPU ---------------------------
        batch   = ds.sample(cfg.batch)               # CPU
        obs_low = _preproc_obs(batch['obs']).to(device)          # (B,356)
        pc      = batch['pointcloud'].to(device)                 # (B,808,6)
        # target = teacher_normalization(batch['teacher_obs'].to(device))          # (B,356)
        target = batch['teacher_obs'].to(device)          # (B,356)

        with torch.no_grad():
            target = teacher_normalization(target)          # (B,356)
            target = obj_encoder(target[:, -16:])
        # target_stored = batch['pc_embedding'].to(device)         # (B,F,D)

        # # -------- (1) update on stored embeddings ---------------------
        # info1 = model.update({'obs': obs_low, 'point_cloud': pc},
        #                      target_stored,
        #                      optimizer=optimiser)
        
        for j in range(3):
            fresh_list = []
            for item in batch['asset']:                              # B items
                mesh_path = os.path.join(
                    "assets/urdf/objects/meshes/custom", item, "textured.obj")
                verts = load_vertices(mesh_path)                     # vertices cached
                emb   = pointnet_embed(verts, pc_encoder)            # NEW embed
                fresh_list.append(emb)
            
            fresh = torch.stack(fresh_list).to(device)               # (B,D)

            obs_dict = {
                'obs': obs_low,
                'point_cloud': pc,
            }

            
            preds = model(obs_dict, fresh)

            # print(f"preds shape: {preds.shape}, target shape: {target.shape}")
            # print("mean value of target:", target.mean().item())
            # print("max value of target:", target.max().item())
            # print("min value of target:", target.min().item())

            loss = nn.MSELoss()(preds, target)  # (B,F,D) vs (B,F,D)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            info2 = {
                'loss': loss.item(),
                'mse': nn.MSELoss()(preds, target).item()
            }


    
        # -------- logging & checkpoint -------------------------------
        if step % cfg.log == 0 or step == 1:
            dt  = time.time() - t0
            fps = (2 * cfg.batch * cfg.frames) / max(dt, 1e-5)   # two updates
            print(f"[{step:>6}/{cfg.steps}] "
                  f"loss2={info2['loss']:.5f}  mse2={info2['mse']:.5f} | "
                  f"fps={fps:,.0f}")
            t0 = time.time()
            torch.save(model.state_dict(), f"adaption_checkpoint_{step:04d}.pt")
            print(f"✓ checkpoint_{step:04d}.pt saved")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data",            help="root folder with *.pt chunks")
    p.add_argument("--out", default="adaption_semantics.pt", help="model checkpoint")
    p.add_argument("--steps",   type=int, default=DEFAULTS['steps'])
    p.add_argument("--batch",   type=int, default=DEFAULTS['batch_size'])
    p.add_argument("--frames",  type=int, default=DEFAULTS['frames_per_ep'])
    p.add_argument("--lr",      type=float, default=DEFAULTS['lr'])
    p.add_argument("--log",     type=int, default=DEFAULTS['log_every'])
    p.add_argument("--hidden_dim", type=int, default=DEFAULTS['hidden_dim'])
    p.add_argument("--repr_dim",   type=int, default=DEFAULTS['repr_dim'])
    p.add_argument("--sem_dim",    type=int, default=DEFAULTS['sem_dim'])
    args = p.parse_args()

    main(args)
