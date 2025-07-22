import einops
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal


from rl_games.common.mlp import MLP

######################################### Deterministic Head #########################################

class TruncatedNormal(D.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist

    def loss_fn(self, dist, target, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

        return {
            "actor_loss": loss,
        }


######################################### Gaussian Mixture Model Head #########################################


class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x, **kwargs):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits
    
    def _time_distributed(self, x, fn):
        """
        Applies `fn` to each time‑step of `x` without an external utility.

        Parameters
        ----------
        x  : torch.Tensor   (B, T, D) or higher‑rank
        fn : callable       expects (B·T, D) → returns tuple of tensors whose
                        first dim is batch.

        Returns
        -------
        Tuple of tensors with shape (B, T, …) each.
        """
        if x.ndim < 3:
            # Nothing to do
            return fn(x)

        B, T = x.shape[:2]
        rest = x.shape[2:]              # feature dims
        y = fn(x.reshape(B * T, *rest))  # flat pass

        # If `fn` returns a single tensor, wrap it in a tuple
        if isinstance(y, torch.Tensor):
            y = (y,)

        # reshape each output back to (B, T, …)
        y = tuple(t.reshape(B, T, *t.shape[1:]) for t in y)
        return y

    def forward(self, x, stddev=None, **kwargs):
        if x.ndim == 3:
            means, scales, logits = self._time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean", **kwargs):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError
        return {
            "actor_loss": loss,
        }


######################################### BeT Head #########################################


class FocalLoss(nn.Module):
    """
    From https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if len(input.shape) == 3:
            N, T, _ = input.shape
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(N, T, 1)).view(N, T)
        elif len(input.shape) == 2:
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BeTHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        nbins=64,
        cluster_centers=None,
        # loss_kwargs
        offset_loss_weight=100.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.cluster_centers = cluster_centers
        self.offset_loss_weight = offset_loss_weight

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        # Bin head
        self.bin_head = nn.Sequential(nn.Linear(hidden_size, nbins))

        # Offset head
        self.repeat_action = hidden_size // self.output_size
        self.offset_head = nn.Sequential(
            nn.Linear(
                hidden_size + self.repeat_action * self.output_size, self.output_size
            )
        )

        # loss
        self.criterion = FocalLoss(gamma=2.0)

    def find_closest_cluster(self, actions, cluster_centers) -> torch.Tensor:
        N, T, _ = actions.shape
        actions = einops.rearrange(actions, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (actions[:, None, :] - cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # N K A -> N K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N )
        closest_cluster_center = closest_cluster_center.view(N, T)
        return closest_cluster_center

    def forward(self, x, stddev=None, cluster_centers=None, **kwargs):
        feat = self.share(x)

        # Bin head
        bin_logits = self.bin_head(feat)

        # get base action
        N, T, choices = bin_logits.shape
        if N > 1:
            # For training, always take the best action
            sampled_center = torch.argmax(bin_logits, dim=-1, keepdim=True)
        else:
            # Sample center based on login probability
            sampled_center = D.Categorical(logits=bin_logits).sample()
        base_action = cluster_centers[sampled_center.flatten()]
        repeated_base_action = base_action.view(N, T, -1).repeat(
            1, 1, self.repeat_action
        )

        # Offset head
        h = torch.cat([feat, repeated_base_action], dim=-1)
        offset = self.offset_head(h)

        return (bin_logits, offset, base_action)

    def loss_fn(self, pred, target, reduction="mean", cluster_centers=None):
        bin_logits, offset, _ = pred

        # Get expert logits and offsets

        true_bins = self.find_closest_cluster(target, cluster_centers)
        true_offsets = target - cluster_centers[true_bins]

        # loss
        discrete_loss = self.criterion(bin_logits, true_bins)
        offset_loss = F.mse_loss(offset, true_offsets)
        actor_loss = discrete_loss + self.offset_loss_weight * offset_loss

        return {
            "actor_loss": actor_loss,
            "discrete_loss": discrete_loss,
            "offset_loss": offset_loss,
        }



######################################### RT-1 Head #########################################


class RT1Head(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        nbins=256,
    ):
        super().__init__()
        self.output_size = output_size
        self.nbins = nbins

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        # Bin head
        self.bin_head = nn.Sequential(nn.Linear(hidden_size, output_size * nbins))

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # initialize action max and min for discretization
        self.action_max, self.action_min = None, None

    def find_closest_cluster(self, actions, cluster_centers) -> torch.Tensor:
        N, T, _ = actions.shape
        actions = einops.rearrange(actions, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (actions[:, None, :] - cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # N K A -> N K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N )
        closest_cluster_center = closest_cluster_center.view(N, T)
        return closest_cluster_center

    def forward(self, x, stddev=None, cluster_centers=None, **kwargs):
        feat = self.share(x)

        # Bin head
        bin_logits = self.bin_head(feat)

        # discretize each action dim
        bin_logits = einops.rearrange(bin_logits, "N T (A K) -> N T A K", K=self.nbins)
        # bin_logits = torch.softmax(bin_logits, dim=-1)

        return self.discrete_to_continuous(bin_logits), bin_logits

    def discretize(self, actions, device):
        actions = torch.tensor(actions)
        self.action_max = torch.max(actions, dim=0)[0].to(device)
        self.action_min = torch.min(actions, dim=0)[0].to(device)

    def discrete_to_continuous(self, action_logits):
        action_logits = torch.argmax(action_logits, dim=-1)
        action_logits = action_logits.float()
        action_logits = (action_logits / (self.nbins - 1)) * (
            self.action_max - self.action_min
        ) + self.action_min
        return action_logits

    def continuous_to_discrete(self, actions):
        actions = (actions - self.action_min) / (self.action_max - self.action_min)
        actions = actions * (self.nbins - 1)
        actions = actions.round()
        return actions

    def loss_fn(self, action, gt_actions, reduction="mean", cluster_centers=None):
        _, action_logits = action

        gt_actions = self.continuous_to_discrete(gt_actions)
        # rearrage for cross entropy loss
        gt_actions = einops.rearrange(gt_actions, "N T A -> (N T) A").long()
        action_logits = einops.rearrange(action_logits, "N T A K -> (N T) K A")

        # loss
        loss = self.criterion(action_logits, gt_actions)

        return {
            "actor_loss": loss,
        }