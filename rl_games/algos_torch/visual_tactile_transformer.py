import einops
import numpy as np
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T

from rl_games.common.rgb_modules import BaseEncoder, ResnetEncoder, weight_init
from rl_games.common.policy_head import (
    DeterministicHead
)
from rl_games.common.gpt import GPT, GPTConfig
from rl_games.common.orderless_gpt import OrderlessGPT, OrderlessGPTConfig
from rl_games.common.mlp import MLP
from rl_games.common.kmeans_discretizer import KMeansDiscretizer

from rl_games.algos_torch.pointnets import PointNet


from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
import torch.distributed as dist


import time
import os


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_type="gpt",
        policy_head="deterministic",
        num_feat_per_step=1,
        device="cuda",
    ):
        super().__init__()

        self._policy_type = policy_type
        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        self._policy = GPT(
            GPTConfig(
                block_size=65,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )
        
        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "gmm":
            self._action_head = GMMHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )

        self.apply(weight_init)


    def forward(self, obs):
        # obs: (B, L, D)   (just frame vectors)
        features = self._policy(obs) 
        return features

class OrderlessActor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_type="gpt",
        policy_head="deterministic",
        num_feat_per_step=1,
        device="cuda",
    ):
        super().__init__()

        self._policy_type = policy_type
        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # OrderlessGPT model
        self._policy = OrderlessGPT(
            OrderlessGPTConfig(
                block_size=65,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )
        
        # if policy_head == "deterministic":
        #     self._action_head = DeterministicHead(
        #         hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
        #     )

        self.apply(weight_init)


    def forward(self, obs):
        # obs: (B, L, D)   (just frame vectors)
        features = self._policy(obs) 
        return features

class ObjectSemanticsTransformer(nn.Module):
    def __init__(self, repr_dim, act_dim, hidden_dim, num_feat_per_step=1):
        super().__init__()
        self._repr_dim = repr_dim + 48 # 48 is the size of the prioperception + tactile stuff
        self._act_dim = act_dim
        self._hidden_dim = hidden_dim + 48
        self._num_feat_per_step = num_feat_per_step

        self.pc_encoder = PointNet(point_channel=6, output_dim=repr_dim) #48 is the size of the prioperception + tactile stuff

        self._transformer = OrderlessActor(
            repr_dim=self._repr_dim,
            act_dim=act_dim,
            hidden_dim=self._hidden_dim,
            policy_type="gpt",
            policy_head="deterministic",
            num_feat_per_step=num_feat_per_step,
        )

        self.semantics_head = DeterministicHead(
                input_size=self._hidden_dim, output_size=self._act_dim, hidden_size=self._hidden_dim, num_layers=2
            )
        


    def forward(self, obs):
        # obs: (B, L, D)  no positional embeddings
        # N is the history length
        low_dim_obs = obs['obs'] 
        # (B, N, 356)
        pc_obs = obs['point_cloud']
        # print("pc_obs shape: ", pc_obs.shape)
        # (B, N, 808, 6)
        B, N, P, C = pc_obs.shape

        pc_obs = pc_obs.view(B * N, P, C)

        #so we dont want to mess with the actual observation size returned by the env,
        # but we can extract the information and build our feature vec

        # the observation is as follows:
        # index range  |  size | description
        # [  0 :  5 ]  #   6   = arm-base joint positions  (always forced to 0.0 for actor)
        # [  6 : 21 ]  #  16   = current finger joint positions (unscaled, ±0.06 noise added)
        # [ 22 : 28 ]  #   7   = **blank / masked** velocity & F-T slots (kept at 0.0)
        # [ 29 : 44 ]  #  16   = previous-target finger positions (unscaled)
        # [ 45 : 60 ]  #  16   = sensed contacts (binary/noisy) → fingertips + selected sensors
        # [ 61 : 84 ]  #  24   = spin-axis helpers (8 fingers × 3-vector, fixed)

        # we only need current finger joint positions, previous-target finger positions, and sensed contacts

        prio_perception = torch.concat( [low_dim_obs[:, :, 6:22], low_dim_obs[:, :, 29:45]], dim=-1)

        binary_contacts = low_dim_obs[:, :, 45:61] # 16 sensed contacts

        pc_feat, _ = self.pc_encoder(pc_obs) # (B* N, repr_dim - 48)
        pc_feat = pc_feat.view(B, N, -1) # (B, N, repr_dim - 48)

        features = torch.concat(
            [prio_perception, binary_contacts, pc_feat], dim=-1
        ).to(low_dim_obs.device) # (B, N, repr_dim)
        # print("features shape: ", features.shape)

        features = self._transformer(features)
        pred_semantics = self.semantics_head(features)
        return pred_semantics.loc


class VTActorCritic(nn.Module):
    def __init__(self, input_shape, repr_dim, act_dim, hidden_dim, num_feat_per_step=1):
        super().__init__()
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self.pc_encoder = PointNet(point_channel=3, output_dim=repr_dim) #48 is the size of the prioperception + tactile stuff

        self._transformer = Actor(
            repr_dim,
            act_dim,
            hidden_dim,
            policy_type="gpt",
            policy_head="deterministic",
            num_feat_per_step=num_feat_per_step,
        )
        self.value_head = self._action_head = DeterministicHead(
                            hidden_dim, 1, hidden_size=hidden_dim, num_layers=2
                        )

        self.semantics_head = self._action_head = DeterministicHead(
                                hidden_dim, 32, hidden_size=hidden_dim, num_layers=2
                            )
        self.action_head = self._action_head = DeterministicHead(
                                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
                            )

    def forward(self, obs, num_prompt_feats, stddev, action=None, cluster_centers=None):

        low_dim_obs = obs['obs'] 
        # (B, 356)
        pc_obs = obs['point_cloud']
        # (B, 808, 6)

        #so we dont want to mess with the actual observation size returned by the env,
        # but we can extract the information and build our feature vec

        # the observation is as follows:
        # index range  |  size | description
        # [  0 :  5 ]  #   6   = arm-base joint positions  (always forced to 0.0 for actor)
        # [  6 : 21 ]  #  16   = current finger joint positions (unscaled, ±0.06 noise added)
        # [ 22 : 28 ]  #   7   = **blank / masked** velocity & F-T slots (kept at 0.0)
        # [ 29 : 44 ]  #  16   = previous-target finger positions (unscaled)
        # [ 45 : 60 ]  #  16   = sensed contacts (binary/noisy) → fingertips + selected sensors
        # [ 61 : 84 ]  #  24   = spin-axis helpers (8 fingers × 3-vector, fixed)

        # we only need current finger joint positions, previous-target finger positions, and sensed contacts

        prio_perception = torch.concat( [low_dim_obs[:, 6:22], low_dim_obs[:, 29:45]], dim=-1, device=low_dim_obs.device).unsqueeze(1)

        binary_contacts = low_dim_obs[:, 45:61] # 16 sensed contacts

        pc_features = self.pc_encoder(pc_obs).unsqueeze(1) # (B, 1, repr_dim - 48)

        features = torch.concat(
            [prio_perception, binary_contacts, pc_features], dim=-1, device=low_dim_obs.device
        ) # (B, N, repr_dim)

        hidden = self._transformer(features)
        pred_value = self.value_head(hidden)
        pred_action = self.action_head(hidden)
        pred_semantics = self.semantics_head(hidden)
        return pred_semantics
    


class VTA2C(a2c_common.A2CBase):
    def __init__(self, base_name, params):
        a2c_common.A2CBase.__init__(self, base_name, params)
        self.is_discrete = False
        self.has_central_value = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.model = VTActorCritic(
            input_shape = self.obs_shape['obs'][0],
            repr_dim=32,
            act_dim=self.actions_num,
            hidden_dim=32, 
            num_feat_per_step=1, 
        )

    def is_rnn(self): return False
    def get_default_rnn_state(self): return []

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, mode='student'):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, mode)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def recon_criterion(self, out, target):
        if self.use_l1:
            return torch.abs(out - target)
        else:
            return (out - target).pow(2)

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        # [HORIZON, OBS_SHAPE], [1, NUM_ENVS * HORIZON / SEQ_LENGTH, HIDDEN_DIM]
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss \
                    = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        # if self.multi_gpu:
        # 	print("====================broadcasting parameters")
        # 	model_params = [self.model.state_dict()]
        # 	dist.broadcast_object_list(model_params, 0)
        # 	self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')
                    if self.game_rewards.current_size > 0:
                        mean_rewards = self.game_rewards.get_mean()
                        mean_lengths = self.game_lengths.get_mean()
                        stats = self.game_rewards.get_stats()
                        print(f"[Rewards] μ={stats['mean'][0]:.3f}  σ={stats['std'][0]:.3f}  "
                            f"min={stats['min'][0]:.3f}  max={stats['max'][0]:.3f}  "
                            f"median={stats['median'][0]:.3f}  n={stats['n']}")
                    print(f'epoch: {epoch_num}, mean rewards: {mean_rewards}, mean lengths: {mean_lengths}')
                    
                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
        