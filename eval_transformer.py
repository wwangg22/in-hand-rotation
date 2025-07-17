
import os, glob, random, torch, collections, gc
import time,argparse, torch.nn as nn, torch.optim as optim


from typing import List, Tuple

from rl_games.algos_torch.visual_tactile_transformer import ObjectSemanticsTransformer

class EpisodeDataset:
    """
    Loads full tensors only when they are actually needed, keeps none.
    """

    def __init__(self, folder: str, pattern="*.pt",
                 recursive=False, min_len=1):
        pat = "**/*.pt" if recursive else pattern
        self.paths = sorted(glob.glob(os.path.join(folder, pat),
                                      recursive=recursive))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {folder}")
        print(f"[EpisodeDataset] found {len(self.paths)} chunk files")

        self.episodes: List[Tuple[int,int,int]] = []   # (cid,start,end)
        self.min_len = max(1, min_len)

        # -------- build lightweight index (only done/env_id) -------------
        for cid, p in enumerate(self.paths):
            done, env_id = self._load_done_env(p)
            self._index_chunk(cid, done, env_id)

        print(f"[EpisodeDataset] indexed {len(self.episodes)} episodes "
              f"(min_len={self.min_len})")

    # ---------------- internal helpers ----------------------------------
    def _load_done_env(self, path):
        tup = torch.load(path, map_location="cpu")
        return tup[-2], tup[-1]          # done, env_id

    def _index_chunk(self, cid, done, env_id):
        order = torch.argsort(env_id, stable=True)
        done, env_id = done[order], env_id[order]
        start = 0
        for count in torch.unique_consecutive(env_id, return_counts=True)[1]:
            stop = start + count
            ep_start = start
            for i in range(start, stop):
                if done[i]:
                    if i - ep_start + 1 >= self.min_len:
                        self.episodes.append((cid, ep_start, i))
                    ep_start = i + 1
            if stop - ep_start >= self.min_len:
                self.episodes.append((cid, ep_start, stop - 1))
            start = stop

    # ---------------- public API ----------------------------------------
    def __len__(self): return len(self.episodes)

    def _sample_one_episode(self, frames):
        # pool = [e for e in self.episodes
        #         if (e[2] - e[1] + 1) >= frames]
        # if not pool:
        #     raise ValueError(f"No episode ≥{frames} frames")
        # cid, s, e = random.choice(pool)
        # idxs = torch.randperm(e - s + 1)[:frames] + s
        # return cid, idxs
        pool = [e for e in self.episodes
            if (e[2] - e[1] + 1) >= frames]
        if not pool:
            raise ValueError(f"No episode ≥{frames} frames")

        cid, s, e = random.choice(pool)     # episode bounds in that chunk
        ep_len   = e - s + 1
        start    = random.randint(0, ep_len - frames)   # window start
        idxs     = torch.arange(start, start + frames) + s   # consecutive
        return cid, idxs

    def _load_chunk(self, cid):
        data = torch.load(self.paths[cid], map_location="cpu")
        return dict(obs=data[0], actions=data[1], sigmas=data[2],
                    pointcloud=data[3], pc_embedding=data[4],
                    done=data[5], env_id=data[6])

    def sample(self, batch_size: int, frames_per_episode: int):
        keys  = ["obs", "actions", "sigmas",
                 "pointcloud", "pc_embedding", "done", "env_id"]
        batch = {k: [] for k in keys}

        choices = [self._sample_one_episode(frames_per_episode)
                   for _ in range(batch_size)]

        by_chunk = {}
        for cid, idxs in choices:
            by_chunk.setdefault(cid, []).append(idxs)

        # -------- load, slice, immediately release ----------------------
        for cid, list_of_idxs in by_chunk.items():
            ch = self._load_chunk(cid)          # load one full chunk
            for idxs in list_of_idxs:
                for k in keys:
                    batch[k].append(ch[k][idxs])
            del ch
            gc.collect()                        # prompt Python to free mem

        for k in keys:
            batch[k] = torch.stack(batch[k], dim=0)
        return batch
    
DEFAULTS = dict(
    hidden_dim      = 80,      # transformer inner size
    repr_dim        = 80,       # point-cloud embedding size
    sem_dim         = 32,       # pc_embedding target size (= act_dim)
    lr              = 1e-4,
    steps           = 400,   # optimisation steps, not epochs
    batch_size      = 256,       # episodes per update
    frames_per_ep   = 12,        # timesteps sampled per episode
    log_every       = 50,
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
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ❶ dataset ----------------------------------------------------------
    ds = EpisodeDataset(
        cfg.data, recursive=True,          # find all .pt under that dir
        min_len=cfg.frames,                # prune short eps
    )

    # ❷ model ------------------------------------------------------------
    model = ObjectSemanticsTransformer(
        repr_dim = cfg.repr_dim,
        act_dim  = cfg.sem_dim,
        hidden_dim = cfg.hidden_dim,
        num_feat_per_step = 1              # you hard-coded this
    ).to(device)

    state_dict = torch.load("./checkpoint_0050.pt", map_location=device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()  # no training, just inference

    # ❸ training loop ----------------------------------------------------
    t0 = time.time()
    with torch.no_grad():
        for step in range(1, cfg.steps + 1):

            batch = ds.sample(cfg.batch, cfg.frames)
            # shapes: (B, F, …)
            obs_low  = _preproc_obs(batch['obs'].to(device))          # (B, F, 356)
            pc       = batch['pointcloud'].to(device)    # (B, F, 808, 6)
            target   = batch['pc_embedding'].to(device)  # (B, F, sem_dim)

            print('mean value in obs_low : ', obs_low.mean().item())

            print("mean value in pc : ", pc.mean().item())

            target = target.mean(dim=1)  # (B, sem_dim)
            print("mean value in target: ", target.mean().item())
            preds = model({'obs': obs_low, 'point_cloud': pc})
            preds = preds.mean(dim=1)  # (B, sem_dim)
            loss = (preds - target).abs().mean()
            print("mean value in preds: ", preds.mean().item())



            print(f"loss={loss.item():.5f}   ")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data",            help="root folder with *.pt chunks")
    p.add_argument("--out", default="semantics.pt", help="model checkpoint")
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
