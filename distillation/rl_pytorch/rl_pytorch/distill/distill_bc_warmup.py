from datetime import datetime
import os
import time
import random

from gym.spaces import Space
import gym
import pickle

import numpy as np
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import central_value
from rl_games.algos_torch import torch_ext

import wandb


class DistillWarmUpTrainer:
    def __init__(
            self,
            teacher_params,
            student_params,
            vec_env,
            num_transitions_per_env,
            num_learning_epochs,
            num_mini_batches,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            init_noise_std=0.05,
            surrogate_loss_coef=1.0,
            value_loss_coef=1.0,
            bc_loss_coef=1.0,
            entropy_coef=0.0,
            use_l1=True,
            learning_rate=1e-3,
            weight_decay=0,
            max_grad_norm=0.5,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=None,
            device='cpu',
            sampler='sequential',
            teacher_log_dir='run',
            student_log_dir='student_run',
            is_testing=False,
            print_log=True,
            apply_reset=False,
            teacher_resume="None",
            vidlogdir='video',
            vid_log_step=1000,
            log_video=False,
            enable_wandb=False,
            bc_warmup=True,
            teacher_data_dir=None,
            worker_id=0,
            warmup_mode=None,
            player=None,
            ablation_mode=None,
            student_resume="None",
    ):
        self.ablation_mode = ablation_mode 

        if not isinstance(vec_env.env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.env.observation_space
        self.action_space = vec_env.env.action_space
        self.state_space = vec_env.env.state_space
        print("Observation space:", self.observation_space)
        print("Action space:", self.action_space)

        self.device = device
        self.desired_kl = desired_kl
        self.writer = SummaryWriter(log_dir=student_log_dir, flush_secs=10)
        self.enable_wandb = enable_wandb

        self.schedule = schedule
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env
        self.clip_param = clip_param
        self.surrogate_loss_coef = surrogate_loss_coef
        self.value_loss_coef = value_loss_coef
        self.bc_loss_coef = bc_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_l1 = use_l1
        self.bc_warmup = bc_warmup

        self.teacher_config = teacher_config = teacher_params['config']
        self.student_config = student_config = student_params['config']
        self.config = teacher_config

        self.num_actors = teacher_config['num_actors']

        self.is_testing = is_testing
        self.vidlogdir = vidlogdir
        self.log_video = log_video
        self.vid_log_step = vid_log_step
        self.rnn_states = None
        self.clip_actions = self.config.get('clip_actions', True)

        # Basic information about environment.
        self.env_info = self.vec_env.get_env_info()
        self.num_agents = self.env_info.get('agents', 1)
        self.normalize_value = self.config.get('normalize_value', False)
        self.normalize_input = teacher_config['normalize_input']
        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.value_size = self.env_info.get('value_size',1)
        self.horizon_length = self.config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.multi_gpu = self.config.get('multi_gpu', False)
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mini_data_size = 4

        self.actions_num = self.action_space.shape[0]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        
        if self.has_central_value:
            if isinstance(self.state_space, gym.spaces.Dict):
                self.state_shape = {}
                for k, v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        # We now define teacher network.
        self.teacher_builder = model_builder.ModelBuilder()
        self.teacher_network = self.teacher_builder.load(teacher_params)
        if isinstance(self.obs_shape, dict):
            self.teacher_obs_shape = self.obs_shape['obs']
        else:
            self.teacher_obs_shape = self.obs_shape

        # print("Teacher observation shape:", self.teacher_obs_shape)
        # print("Observation space:", self.observation_space) 
        # exit()
        self.teacher_build_config = {
            'actions_num': self.actions_num,
            'input_shape': self.teacher_obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.teacher_actor_critic = self.teacher_network.build(self.teacher_build_config)
        self.teacher_actor_critic.to(self.device)
        
        if self.has_central_value:
            print('Adding Central Value Network')
            from omegaconf import open_dict
            if 'model' not in self.config['central_value_config']:
                with open_dict(self.config):
                    self.config['central_value_config']['model'] = {'name': 'central_value'}
            builder = model_builder.ModelBuilder()
            tea_network = builder.load(self.config['central_value_config'])
            teacher_cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'normalize_value': self.normalize_value,
                'network': tea_network,
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu,
            }
            self.teacher_central_value_net = central_value.CentralValueTrain(**teacher_cv_config).to(self.device)

        # We now define student network.
        self.student_builder = model_builder.ModelBuilder()
        self.student_network = self.student_builder.load(student_params)
        if self.ablation_mode == "no-tactile":
            self.student_obs_shape = {'obs': (276, ), 'pointcloud': (680, 6)}
        elif self.ablation_mode == "multi-modality-plus":
            self.student_obs_shape = {'obs': self.obs_shape['student_obs'], 'pointcloud': (808, 6)}
        elif self.ablation_mode == "aug":
            self.student_obs_shape = {'obs': self.obs_shape['student_obs'], 'pointcloud': (680, 6)}
        elif self.ablation_mode == "no-pc":
            self.student_obs_shape = self.obs_shape['student_obs']
        else:
            raise NotImplementedError
        
        self.student_build_config = {
            'actions_num': self.actions_num,
            'input_shape': self.student_obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.student_actor_critic = self.student_network.build(self.student_build_config)
        self.student_actor_critic.to(self.device)

        if student_resume is not None and student_resume != "None":
            student_path = "{}/{}.pth".format(student_log_dir, student_resume)
            print("Loading student model from", student_path)
            self.student_load(student_path)

        self.optimizer = optim.Adam(
            self.student_actor_critic.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # PPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm

        # Log
        self.teacher_log_dir = teacher_log_dir
        # student Log
        self.student_log_dir = student_log_dir
        self.print_log = print_log
        
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset
        self.teacher_resume = teacher_resume
        self.storage_load_path = "./teacher_data.pkl"
        assert teacher_resume is not None
        self.teacher_data_dir = teacher_data_dir

    def mini_batch_generator(self, num_mini_batches):
        batch_size = 200 * 64 * self.mini_data_size  # self.vec_env.env.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
        # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
        subset = SequentialSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
    
    def test_teacher(self, path):
        self.teacher_actor_critic.load_state_dict(torch.load(path))
        self.teacher_actor_critic.eval()

    def test_student(self, path):
        self.student_actor_critic.load_state_dict(torch.load(path))
        self.student_actor_critic.eval()

    def teacher_load(self, path):
        checkpoint = torch_ext.load_checkpoint(path)
        self.teacher_actor_critic.load_state_dict(checkpoint['model'])
        self.set_stats_weights(self.teacher_actor_critic, checkpoint)
        if self.has_central_value:
            self.teacher_central_value_net.load_state_dict(checkpoint['assymetric_vf_nets'])
            self.teacher_central_value_net.eval()
        env_state = checkpoint.get('env_state', None)
        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)
        self.teacher_actor_critic.eval()

    def student_load(self, path):
        print("Loading student model from", path)
        checkpoint = torch_ext.load_checkpoint(path)
        self.student_actor_critic.load_state_dict(checkpoint['model'])
        self.set_stats_weights(self.student_actor_critic, checkpoint)
        env_state = checkpoint.get('env_state', None)
        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

        self.student_actor_critic.train()

    def set_stats_weights(self, model, weights):
        if self.normalize_input and 'running_mean_std' in weights:
            model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            model.value_mean_std.load_state_dict(weights['reward_mean_std'])

    def get_weights(self):
        state = {}
        state['model'] = self.student_actor_critic.state_dict()
        return state

    def get_full_state_weights(self):
        state = self.get_weights()
        state['optimizer'] = self.optimizer.state_dict()

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def save(self, path):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(path, state)

    def _preproc_obs(self, obs_batch):
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

    def get_teacher_central_value(self, obs_dict):
        return self.teacher_central_value_net.get_value(obs_dict)

    def get_action_values(self, model, obs, mode='teacher'):
        processed_obs = self._preproc_obs(obs['obs'])

        model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }

        with torch.no_grad():
            res_dict = model(input_dict)
            if self.has_central_value and mode == 'teacher':
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                if mode == 'teacher':
                    value = self.get_teacher_central_value(input_dict)
                else:
                    raise NotImplementedError
                res_dict['values'] = value

        return res_dict

    def calc_gradients(self, model, input_dict, prev_actions):
        model.train()

        obs_batch = self._preproc_obs(input_dict)
        batch_dict = {
                'is_train': True,
                'prev_actions': prev_actions,
                'obs': obs_batch,
        }

        res_dict = model(batch_dict)
        return res_dict
    
    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.env.get_state()

        self.num_teacher_transitions = 102400000
        self.num_batch = 125 

        for it in range(self.current_learning_iteration, num_learning_iterations):
            # Learning step
            start = time.time()

            mean_loss = self.update(it)

            stop = time.time()
            learn_time = stop - start
            if self.print_log:
                self.log(locals())
            if it % log_interval == 0:
                self.save(os.path.join(self.student_log_dir,
                                        'model_bc_{}'.format(it)))

        self.save(os.path.join(self.student_log_dir,
                                   'model_bc_{}'.format(num_learning_iterations)))
    def dagger_run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.env.get_state()

        self.num_teacher_transitions = 102400000
        self.num_batch = 125 
        self.dagger_it = -1
        dagger_update_interval = 2000

        for it in range(self.current_learning_iteration, num_learning_iterations):
            # Learning step
            start = time.time()

            mean_loss = self.dagger_update(it, self.dagger_it)

            stop = time.time()
            learn_time = stop - start
            if self.print_log:
                self.log(locals())
            if it % log_interval == 0:
                self.save(os.path.join(self.student_log_dir,
                                        'model_bc_{}'.format(it)))
            if it % dagger_update_interval == 0 and it != 0:
                print("DAGGER UPDATE")
                self.dagger_it += 1
                self.collect_dagger_rollouts(horizon=10, dagger_it=self.dagger_it)

        self.save(os.path.join(self.student_log_dir,
                                   'model_bc_{}'.format(num_learning_iterations)))
        
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.env.num_envs
        self.tot_time += locs['learn_time']
        iteration_time = locs['learn_time']

        ep_string = f''

        self.writer.add_scalar('Loss/BC',
                               locs['mean_loss'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.env.num_envs /
                  (locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Total loss:':>{pad}} {locs['mean_loss']:.4f}\n""")
        if self.enable_wandb:
            wandb.log({
                'Total loss:': locs['mean_loss']
            })

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def recon_criterion(self, out, target):
        if self.use_l1:
            return torch.abs(out - target)
        else:
            return (out - target).pow(2)

    def update(self, it):
        mean_bc_loss = 0

        batch = self.mini_batch_generator(self.num_mini_batches)
        teacher_obs_storage, teacher_actions_storage, teacher_sigmas_storage, teacher_pointcloud_storage = [], [], [], []
        teacher_file_list = os.listdir(self.teacher_data_dir)
        for idx_worker in range(self.mini_data_size):
            load_dir = os.path.join(self.teacher_data_dir, random.choice(teacher_file_list)) 
            print(load_dir)
            teacher_obs_tmp, teacher_actions_tmp, teacher_sigmas_tmp, teacher_pointcloud_tmp = torch.load(load_dir) 
            teacher_obs_storage.extend(teacher_obs_tmp.cpu())
            teacher_actions_storage.extend(teacher_actions_tmp.cpu())
            # teacher_expert_obs_storage.extend(teacher_expert_obs_tmp.cpu())
            teacher_sigmas_storage.extend(teacher_sigmas_tmp.cpu())
            teacher_pointcloud_storage.extend(teacher_pointcloud_tmp.cpu())

        teacher_obs_storage = torch.stack(teacher_obs_storage, dim=0)
        teacher_actions_storage = torch.stack(teacher_actions_storage, dim=0)
        # teacher_expert_obs_storage = torch.stack(teacher_expert_obs_storage, dim=0)
        teacher_sigmas_storage = torch.stack(teacher_sigmas_storage, dim=0)
        teacher_pointcloud_storage = torch.stack(teacher_pointcloud_storage, dim=0)
        
        if self.ablation_mode == "no-tactile":
            teacher_obs_storage = torch.cat([teacher_obs_storage[:, :45], teacher_obs_storage[:, 61:130],
                                                teacher_obs_storage[:, 146:215], teacher_obs_storage[:, 231:300], 
                                                teacher_obs_storage[:, 316:]], dim=-1)
            teacher_pointcloud_storage = teacher_pointcloud_storage[:, :680, :]
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape, teacher_pointcloud_storage.shape)
        elif self.ablation_mode == "multi-modality-plus":
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape, teacher_pointcloud_storage.shape)
        elif self.ablation_mode == "aug":
            teacher_pointcloud_storage = teacher_pointcloud_storage[:, :680, :]
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape, teacher_pointcloud_storage.shape)
        elif self.ablation_mode == "no-pc":
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape)
        else:
            raise NotImplementedError

        self.num_learning_epochs = 1
        for epoch in range(self.num_learning_epochs):
            print("UPDATE START EPOCH", epoch)
            for batch_idx, indices in enumerate(batch):
                teacher_obs_batch = teacher_obs_storage[indices].to(self.device)
                teacher_actions_batch = teacher_actions_storage[indices].to(self.device)
                # teacher_expert_obs_batch = teacher_expert_obs_storage[indices].to(self.device)
                teacher_sigmas_batch = teacher_sigmas_storage[indices].to(self.device)
                if self.ablation_mode != "no-pc": 
                    teacher_pointcloud_batch = teacher_pointcloud_storage[indices].to(self.device)
                    student_obs_batch = {'obs': teacher_obs_batch, 'pointcloud': teacher_pointcloud_batch}
                    if teacher_pointcloud_batch.shape[1] == 808:
                        for i in range(teacher_pointcloud_batch.shape[0]):  # remove padded points
                            zeros = torch.where(teacher_pointcloud_batch[i, :, :3] == -torch.tensor([5.7225e-01, 1.0681e-04, 1.7850e-01]).cuda())
                            teacher_pointcloud_batch[i][zeros[0]] = teacher_pointcloud_batch[i, 0, :].clone()
                            zeros2 = torch.where(teacher_pointcloud_batch[i, :, :3] == -torch.tensor([5.3432e-01, -1.5243e-06, 2.0256e-01]).cuda())
                            teacher_pointcloud_batch[i][zeros2[0]] = teacher_pointcloud_batch[i, 0, :].clone()
                else:
                    student_obs_batch = teacher_obs_batch
                # with torch.no_grad():
                #     tmp_obs = {
                #         'obs': teacher_expert_obs_batch,}
                #     teacher_res_dict = self.get_action_values(self.teacher_actor_critic, tmp_obs, mode='teacher')
                #     teacher_actions = res_dict['actions']
                
                #print mean difference between teacher actions and teachers expert actions batch
                # print("Teacher actions mean difference:", torch.abs(teacher_actions - teacher_actions_batch).mean().item())
                
                res_dict = self.calc_gradients(self.student_actor_critic, student_obs_batch, teacher_actions_batch)

                mu_batch = res_dict['mus']
                sigma_batch = res_dict['sigmas']


                # print("std of teacher actions:", torch.clamp(teacher_actions_batch, -1.0, 1.0).std())
                # print("average of teacher actions:", torch.clamp(teacher_actions_batch, -1.0, 1.0).mean())

                # print("std of student actions:", mu_batch.std())
                # print("average of student actions:", mu_batch.mean())

                # close_mask   = teacher_actions_batch.abs() >= 0.99            # bool tensor
                # num_close    = close_mask.sum().item()          # total saturated scalars
                # total_vals   = teacher_actions_batch.numel()
                # pct_close    = 100.0 * num_close / total_vals   # percentage

                # print(f"{num_close:,}/{total_vals:,} "
                #     f"values are within ±0.05 of the rail  "
                #     f"({pct_close:.1f} %).")

                # Imitation loss
                bc_loss = torch.sum(self.recon_criterion(mu_batch, torch.clamp(teacher_actions_batch, -1.0, 1.0)), dim=-1).mean() 
                loss = self.bc_loss_coef * bc_loss
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                        self.student_actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_bc_loss += bc_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_bc_loss /= num_updates

        return mean_bc_loss
    
    def dagger_update(self, it, cur_dag_it):
        mean_bc_loss = 0

        batch = self.mini_batch_generator(self.num_mini_batches)
        teacher_obs_storage, teacher_actions_storage, teacher_sigmas_storage, teacher_pointcloud_storage = [], [], [], []
        teacher_file_list = os.listdir(self.teacher_data_dir)
        fresh_dagger_file_list = [f for f in teacher_file_list if f.startswith(f"dagger_{cur_dag_it}_")]
        #new list without fresh dagger files
        teacher_file_list = [f for f in teacher_file_list if not f.startswith(f"dagger_{cur_dag_it}_")]
        for idx_worker in range(self.mini_data_size):
            if idx_worker < self.mini_data_size // 2 and  len(fresh_dagger_file_list) > 0:
                load_dir = os.path.join(self.teacher_data_dir, random.choice(fresh_dagger_file_list))
            else:
                load_dir = os.path.join(self.teacher_data_dir, random.choice(teacher_file_list)) 
            teacher_obs_tmp, teacher_actions_tmp, teacher_sigmas_tmp, teacher_pointcloud_tmp = torch.load(load_dir) 
            teacher_obs_storage.extend(teacher_obs_tmp.cpu())
            teacher_actions_storage.extend(teacher_actions_tmp.cpu())
            # teacher_expert_obs_storage.extend(teacher_expert_obs_tmp.cpu())
            teacher_sigmas_storage.extend(teacher_sigmas_tmp.cpu())
            teacher_pointcloud_storage.extend(teacher_pointcloud_tmp.cpu())

        teacher_obs_storage = torch.stack(teacher_obs_storage, dim=0)
        teacher_actions_storage = torch.stack(teacher_actions_storage, dim=0)
        # teacher_expert_obs_storage = torch.stack(teacher_expert_obs_storage, dim=0)
        teacher_sigmas_storage = torch.stack(teacher_sigmas_storage, dim=0)
        teacher_pointcloud_storage = torch.stack(teacher_pointcloud_storage, dim=0)
        
        if self.ablation_mode == "no-tactile":
            teacher_obs_storage = torch.cat([teacher_obs_storage[:, :45], teacher_obs_storage[:, 61:130],
                                                teacher_obs_storage[:, 146:215], teacher_obs_storage[:, 231:300], 
                                                teacher_obs_storage[:, 316:]], dim=-1)
            teacher_pointcloud_storage = teacher_pointcloud_storage[:, :680, :]
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape, teacher_pointcloud_storage.shape)
        elif self.ablation_mode == "multi-modality-plus":
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape, teacher_pointcloud_storage.shape)
        elif self.ablation_mode == "aug":
            teacher_pointcloud_storage = teacher_pointcloud_storage[:, :680, :]
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape, teacher_pointcloud_storage.shape)
        elif self.ablation_mode == "no-pc":
            print(teacher_obs_storage.shape, teacher_actions_storage.shape, teacher_sigmas_storage.shape)
        else:
            raise NotImplementedError

        self.num_learning_epochs = 1
        for epoch in range(self.num_learning_epochs):
            print("UPDATE START EPOCH", epoch)
            for batch_idx, indices in enumerate(batch):
                teacher_obs_batch = teacher_obs_storage[indices].to(self.device)
                teacher_actions_batch = teacher_actions_storage[indices].to(self.device)
                # teacher_expert_obs_batch = teacher_expert_obs_storage[indices].to(self.device)
                teacher_sigmas_batch = teacher_sigmas_storage[indices].to(self.device)
                if self.ablation_mode != "no-pc": 
                    teacher_pointcloud_batch = teacher_pointcloud_storage[indices].to(self.device)
                    student_obs_batch = {'obs': teacher_obs_batch, 'pointcloud': teacher_pointcloud_batch}
                    if teacher_pointcloud_batch.shape[1] == 808:
                        for i in range(teacher_pointcloud_batch.shape[0]):  # remove padded points
                            zeros = torch.where(teacher_pointcloud_batch[i, :, :3] == -torch.tensor([5.7225e-01, 1.0681e-04, 1.7850e-01]).cuda())
                            teacher_pointcloud_batch[i][zeros[0]] = teacher_pointcloud_batch[i, 0, :].clone()
                            zeros2 = torch.where(teacher_pointcloud_batch[i, :, :3] == -torch.tensor([5.3432e-01, -1.5243e-06, 2.0256e-01]).cuda())
                            teacher_pointcloud_batch[i][zeros2[0]] = teacher_pointcloud_batch[i, 0, :].clone()
                else:
                    student_obs_batch = teacher_obs_batch
                # with torch.no_grad():
                #     tmp_obs = {
                #         'obs': teacher_expert_obs_batch,}
                #     teacher_res_dict = self.get_action_values(self.teacher_actor_critic, tmp_obs, mode='teacher')
                #     teacher_actions = res_dict['actions']
                
                #print mean difference between teacher actions and teachers expert actions batch
                # print("Teacher actions mean difference:", torch.abs(teacher_actions - teacher_actions_batch).mean().item())
                
                res_dict = self.calc_gradients(self.student_actor_critic, student_obs_batch, teacher_actions_batch)

                mu_batch = res_dict['mus']
                sigma_batch = res_dict['sigmas']


                # print("std of teacher actions:", torch.clamp(teacher_actions_batch, -1.0, 1.0).std())
                # print("average of teacher actions:", torch.clamp(teacher_actions_batch, -1.0, 1.0).mean())

                # print("std of student actions:", mu_batch.std())
                # print("average of student actions:", mu_batch.mean())

                # close_mask   = teacher_actions_batch.abs() >= 0.99            # bool tensor
                # num_close    = close_mask.sum().item()          # total saturated scalars
                # total_vals   = teacher_actions_batch.numel()
                # pct_close    = 100.0 * num_close / total_vals   # percentage

                # print(f"{num_close:,}/{total_vals:,} "
                #     f"values are within ±0.05 of the rail  "
                #     f"({pct_close:.1f} %).")

                # Imitation loss
                bc_loss = torch.sum(self.recon_criterion(mu_batch, torch.clamp(teacher_actions_batch, -1.0, 1.0)), dim=-1).mean() 
                loss = self.bc_loss_coef * bc_loss
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                        self.student_actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_bc_loss += bc_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_bc_loss /= num_updates

        return mean_bc_loss

    @torch.no_grad()
    def evaluate_student(self,
                        n_spots: int = 500,
                        render: bool = False):

        assert self.vec_env is not None, "vec_env is required for evaluation"
        self.student_actor_critic.eval()

        n_envs          = self.vec_env.env.num_envs
        obs_env         = self.vec_env.reset()
        running_return  = np.zeros(n_envs, dtype=np.float32)
        step_counter    = np.zeros(n_envs, dtype=np.int32)      # NEW – per-env step counts
        finished_mask   = np.zeros(n_envs, dtype=bool)          # already ensures one-shot logging

        ep_returns: list[float] = []
        ep_lengths: list[int]   = []

        while not finished_mask.all():                          # stop when every env finished once
            # ---------- build observations exactly as during training ----------
            if self.ablation_mode != "no-pc":
                student_obs = {
                    'obs': torch.as_tensor(obs_env['obs']['student_obs'], device=self.device),
                    'pointcloud': torch.as_tensor(obs_env['obs']['pointcloud'], device=self.device)
                }
            else:
                student_obs = torch.as_tensor(obs_env['obs']['student_obs'], device=self.device)

            student_obs = self._preproc_obs(student_obs)

            inp = {
                'is_train'   : False,
                'prev_actions': None,
                'obs'        : student_obs,
                'rnn_states' : self.rnn_states
            }

            actions = self.student_actor_critic(inp)['actions']
            actions = torch.clamp(actions, -1.0, 1.0)

            # Send dummy zero actions to envs that have already finished
            if finished_mask.any():
                actions[finished_mask] = 0.0

            # ------------------- environment step -----------------------------
            obs_env, rew, done, _ = self.vec_env.step(actions)

            rew_np  = rew.cpu().numpy()  if torch.is_tensor(rew)  else rew
            done_np = done.cpu().numpy() if torch.is_tensor(done) else done

            # count this step for unfinished envs
            step_counter[~finished_mask] += 1
            running_return += rew_np.squeeze()

            # on first completion per env, record return & length
            for idx in range(n_envs):
                if done_np[idx] and not finished_mask[idx]:
                    ep_returns.append(float(running_return[idx]))
                    ep_lengths.append(int(step_counter[idx]))
                    finished_mask[idx]  = True
                    running_return[idx] = 0.0          # stop accumulating
                    step_counter[idx]   = 0            # (optional) stop counting

            if render and hasattr(self.vec_env, 'render'):
                self.vec_env.render(mode='human')

        # ------------------------- statistics ---------------------------------
        mean_ret = np.mean(ep_returns)
        std_ret  = np.std(ep_returns)
        max_ret  = np.max(ep_returns)
        min_ret  = np.min(ep_returns)

        mean_len = np.mean(ep_lengths)
        std_len  = np.std(ep_lengths)
        max_len  = np.max(ep_lengths)
        min_len  = np.min(ep_lengths)

        print(f"Student policy over {len(ep_returns)} episodes:")
        print(f"  return | mean: {mean_ret:.2f}  std: {std_ret:.2f}  min: {min_ret:.2f}  max: {max_ret:.2f}")
        print(f"   steps | mean: {mean_len:.1f}  std: {std_len:.1f}  min: {min_len}      max: {max_len}")

        # keep original signature (first element) but also give lengths if caller wants them
        return ep_returns, ep_lengths


    @torch.no_grad()
    def evaluate_teacher(self,
                        render=  False):
        """
        Run one evaluation pass with the **expert / teacher** policy.

        Stops after every parallel env finishes a single episode.

        Returns
        -------
        (ep_returns, ep_lengths)
            Lists of per-episode returns and lengths, in env-index order.
        """
        assert self.vec_env is not None, "`vec_env` must be initialised"

        # ─── make sure weights are loaded & switch to eval ────────────────────────
        self.teacher_load(f"{self.teacher_log_dir}/{self.teacher_resume}.pth")
        self.teacher_actor_critic.eval()

        n_envs          = self.vec_env.env.num_envs
        obs_env         = self.vec_env.reset()

        running_return  = np.zeros(n_envs, dtype=np.float32)
        step_counter    = np.zeros(n_envs, dtype=np.int32)
        finished_mask   = np.zeros(n_envs, dtype=bool)

        ep_returns: list[float] = []
        ep_lengths: list[int]   = []

        while not finished_mask.all():
            # ───── build teacher input exactly as in training ────────────────────
            teacher_obs          = obs_env.copy()
            teacher_obs["obs"]   = obs_env["obs"]["obs"]       # unwrap dict-of-dict
            res                  = self.get_action_values(self.teacher_actor_critic,
                                                        teacher_obs,
                                                        mode="teacher")
            actions              = torch.clamp(res["actions"], -1.0, 1.0)

            # freeze finished envs
            if finished_mask.any():
                actions[finished_mask] = 0.0

            # ───── environment step ──────────────────────────────────────────────
            obs_env, rew, done, _ = self.vec_env.step(actions)

            rew_np   = rew.cpu().numpy()   if torch.is_tensor(rew)   else rew
            done_np  = done.cpu().numpy()  if torch.is_tensor(done)  else done

            step_counter[~finished_mask] += 1
            running_return               += rew_np.squeeze()

            # first completion per env → log stats
            for idx in range(n_envs):
                if done_np[idx] and not finished_mask[idx]:
                    ep_returns.append(float(running_return[idx]))
                    ep_lengths.append(int(step_counter[idx]))
                    finished_mask[idx]  = True
                    running_return[idx] = 0.0
                    step_counter[idx]   = 0

            if render and hasattr(self.vec_env, "render"):
                self.vec_env.render(mode="human")

        # ────────── summary ───────────────────────────────────────────────────────
        mean_ret, std_ret = np.mean(ep_returns), np.std(ep_returns)
        max_ret,  min_ret = np.max(ep_returns), np.min(ep_returns)
        mean_len, std_len = np.mean(ep_lengths), np.std(ep_lengths)
        max_len,  min_len = np.max(ep_lengths), np.min(ep_lengths)

        print(f"Teacher policy over {len(ep_returns)} episodes:")
        print(f"  return | mean: {mean_ret:.2f}  std: {std_ret:.2f}  "
            f"min: {min_ret:.2f}  max: {max_ret:.2f}")
        print(f"   steps | mean: {mean_len:.1f}  std: {std_len:.1f}  "
            f"min: {min_len}      max: {max_len}")

        return ep_returns, ep_lengths

    @torch.no_grad()
    def collect_dagger_rollouts(self,
                                horizon: int = 5000,
                                save_every: int = 200,
                                dagger_it: int = 0):
        """
        DAgger collection that matches DistillCollector.run storage format:
            tuple(obs, teacher_mus, teacher_sigmas, pointcloud)
        Each .pt file holds (num_envs * 200, 4, …) rows.
        The *student* policy drives the env; the *teacher* labels each state.
        """
        # ───────────────────── setup ───────────────────────────────────────────

        current_obs = self.vec_env.reset()
        current_states = self.vec_env.env.get_state()

        self.teacher_load(
            "{}/{}.pth".format(self.teacher_log_dir, self.teacher_resume))
        cur_reward_sum = torch.zeros(
            self.vec_env.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.vec_env.env.num_envs, dtype=torch.float, device=self.device)

        reward_sum = []
        episode_length = []

        for it in range(0, horizon):
            # report_gpu()
            ep_infos = []

            storage = {'obs': [], 'actions': [], 'sigmas': [], 'pointcloud': []}  # , 'pointcloud': []}

            # Rollout
            for i in range(200):
                if i % 100 == 99:
                    print(i)
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                    current_states = self.vec_env.get_state()
                
                teacher_obs = current_obs.copy()
                teacher_obs["obs"] = current_obs["obs"]["obs"]
                with torch.no_grad():
                    if self.ablation_mode != "no-pc":
                        student_obs = {
                            'obs'       : torch.as_tensor(current_obs['obs']['student_obs'],
                                                        device=self.device),
                            'pointcloud': torch.as_tensor(current_obs['obs']['pointcloud'],
                                                        device=self.device)
                        }
                    else:
                        student_obs = torch.as_tensor(current_obs['obs']['student_obs'],
                                                    device=self.device)
                    student_obs = self._preproc_obs(student_obs)


                    inp = {'is_train'   : False,
                        'prev_actions': None,
                        'obs'         : student_obs,
                        'rnn_states'  : self.rnn_states}
                    student_actions = torch.clamp(
                                        self.student_actor_critic(inp)['actions'],
                                        -1.0, 1.0) 
                
                # Compute the action
                with torch.no_grad():
                    res_dict = self.get_action_values(self.teacher_actor_critic, teacher_obs, mode='teacher')
                    teacher_actions = res_dict['actions']
                    teacher_mus = res_dict['mus']
                    teacher_sigmas = res_dict['sigmas']

                    storage['obs'].extend(current_obs['obs']['student_obs'])
                    storage['actions'].extend(teacher_mus)
                    storage['sigmas'].extend(teacher_sigmas)
                    storage['pointcloud'].extend(current_obs['obs']['pointcloud'])

                    next_obs, rews, dones, infos = self.vec_env.step(torch.clamp(student_actions, - 1.0, 1.0))
                    next_states = self.vec_env.env.get_state()
                # Record the transition
                current_obs = next_obs
                current_states.copy_(next_states)
                ep_infos.append(infos)

                if self.print_log:
                    cur_reward_sum[:] += rews
                    cur_episode_length[:] += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    reward_sum.extend(
                        cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    episode_length.extend(
                        cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                    if i % 100 == 99:
                        print(np.mean(reward_sum), np.mean(episode_length))
                if i % 200 == 199:
                    for key in storage.keys():
                        storage[key] = torch.stack(storage[key], dim=0)
                        print(storage[key].shape)
                    save_dir = os.path.join(self.teacher_data_dir, "dagger_{}_{}_{}.pt".format(dagger_it, it, int((i-199)/200)))
                    torch.save((storage['obs'], storage['actions'], storage['sigmas'], storage['pointcloud']), save_dir)  
                    storage = {'obs': [], 'actions': [], 'sigmas': [], 'pointcloud': []} 
                    reward_sum = []
                    episode_length = []


        # print("Loading teacher model from",
        #     f"{self.teacher_log_dir}/{self.teacher_resume}.pth")
        # self.teacher_load(f"{self.teacher_log_dir}/{self.teacher_resume}.pth")

        # if horizon is None:
        #     horizon = self.num_transitions_per_env          # default = 200 in run()

        # save_dir = os.path.join(self.teacher_data_dir)
        # os.makedirs(save_dir, exist_ok=True)

        # storage = {'obs': [],'actions': [], 'sigmas': [], 'pointcloud': []}
        # batch_idx      = 0
        # steps_in_batch = 0

        # self.student_actor_critic.eval()
        # obs_env = self.vec_env.reset()

        # for step in range(horizon):
        #     teacher_obs        = obs_env.copy()
        #     teacher_obs['obs'] = obs_env['obs']['obs']           # identical to run()
        #     res_teacher        = self.get_action_values(
        #                             self.teacher_actor_critic,
        #                             teacher_obs, mode='teacher')

        #     teacher_mus    = res_teacher['mus']       
        #     teacher_sigmas = res_teacher['sigmas']    
        #     teacher_actions = res_teacher['actions']  

        #     if self.ablation_mode != "no-pc":
        #         student_obs = {
        #             'obs'       : torch.as_tensor(obs_env['obs']['student_obs'],
        #                                         device=self.device),
        #             'pointcloud': torch.as_tensor(obs_env['obs']['pointcloud'],
        #                                         device=self.device)
        #         }
        #     else:
        #         student_obs = torch.as_tensor(obs_env['obs']['student_obs'],
        #                                     device=self.device)
        #     student_obs = self._preproc_obs(student_obs)

           

        #     inp = {'is_train'   : False,
        #         'prev_actions': None,
        #         'obs'         : student_obs,
        #         'rnn_states'  : self.rnn_states}
        #     student_actions = torch.clamp(
        #                         self.student_actor_critic(inp)['actions'],
        #                         -1.0, 1.0)

        #     storage['obs'       ].extend(obs_env['obs']['student_obs'])
        #     # storage['expert_obs'].extend(obs_env['obs']['obs'])
        #     storage['actions'   ].extend(teacher_actions.cpu())
        #     storage['sigmas'    ].extend(teacher_sigmas.cpu())
        #     storage['pointcloud'].extend(obs_env['obs']['pointcloud'])

        #     obs_env, _, _, _ = self.vec_env.step(torch.clamp(teacher_actions, -1.0, 1.0))

        #     steps_in_batch += 1
        #     if steps_in_batch == save_every:
        #         for k in storage:
        #             storage[k] = torch.stack(storage[k], dim=0)  # keep (N_envs*200,4,…)
        #             print(f"{k:>10} batch shape → {tuple(storage[k].shape)}")

        #         file_name = f"dagger_batch_{3}_{batch_idx}.pt"
        #         torch.save((storage['obs'],
        #                     storage['actions'],
        #                     # storage['expert_obs'],
        #                     storage['sigmas'],
        #                     storage['pointcloud']),
        #                 os.path.join(save_dir, file_name))
        #         print(f"  › saved {file_name}")

        #         storage        = {'obs': [], 'actions': [], 'sigmas': [], 'pointcloud': []}
        #         steps_in_batch = 0
        #         batch_idx     += 1
