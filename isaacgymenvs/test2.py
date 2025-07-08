"""
Modified train.py for demo‑follow view (v5)
-------------------------------------------
* Removed runner‑level monkey‑patch (player wasn’t created yet → NoneType).
* **Recording is now injected directly into the env** inside `create_env_thunk`:
  we wrap `envs.step` and `envs.reset` to grab a frame, so it works regardless
  of when RL‑Games constructs its `Player`.
"""

import datetime, random, os, sys, hydra, yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgym import gymapi, gymtorch
import numpy as np
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.reformat import omegaconf_to_dict, print_dict
from utils.utils import set_np_formatting, set_seed

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import (
        amp_continuous, amp_players, amp_models, amp_network_builder,
    )
    import isaacgymenvs

    # ────────────────── Boilerplate overrides ──────────────────
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    if cfg.test:
        cfg.task.env.numEnvs = 1
        # Respect whatever the caller set for headless. If they passed
        # headless=True we stay fully off‑screen; otherwise the viewer opens.
        # Only force_render when the viewer is active.
        if not cfg.headless:
            cfg.force_render = True

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        if not os.path.isfile(cfg.checkpoint):
            raise FileNotFoundError(cfg.checkpoint)
        print("[INFO] Using checkpoint:", cfg.checkpoint)

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # ────────────────── Env factory with built‑in recorder ──────────────────
    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

        if not cfg.headless:
            eye, target = gymapi.Vec3(1.2, 1.2, 1.0), gymapi.Vec3(0, 0, 0)
            envs.gym.viewer_camera_look_at(envs.viewer, envs.envs[0], eye, target)

        # ───── Manual video capture baked into env methods ─────
        if cfg.capture_video:
            video_dir = os.path.join("videos", run_name)
            os.makedirs(video_dir, exist_ok=True)
            writer = imageio.get_writer(os.path.join(video_dir, "demo.mp4"), fps=30)

            cam_props = gymapi.CameraProperties()
            cam_props.width, cam_props.height = 720, 720
                                    # create camera attached to env‑0 *root rigid body*
            env_ptr = envs.envs[0]
            cam_handle = envs.gym.create_camera_sensor(env_ptr, cam_props)

                        # actor 0, rigid body 0 is usually the root
            root_body = envs.gym.get_actor_rigid_body_handle(env_ptr, 0, 0)
            trans = gymapi.Transform()  # identity (no offset)
            envs.gym.attach_camera_to_body(cam_handle, env_ptr, root_body, trans, gymapi.FOLLOW_POSITION)

            def grab_frame():
                envs.gym.render_all_camera_sensors(envs.sim)
                buf = envs.gym.get_camera_image(envs.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                if buf is None or buf.size == 0:
                    return  # skip if frame not yet ready (e.g., right after reset)
                im = np.reshape(buf, (cam_props.height, cam_props.width, 4))[:, :, :3]
                writer.append_data(im)

            # Patch step and reset
            orig_step, orig_reset = envs.step, envs.reset

            def step_wrapped(action):
                obs, rew, done, info = orig_step(action)
                grab_frame()
                return obs, rew, done, info

            def reset_wrapped(*a, **k):
                obs = orig_reset(*a, **k)
                grab_frame()
                return obs

            envs.step = step_wrapped
            envs.reset = reset_wrapped
            envs._video_writer = writer  # store for closing later

        return envs

    # Register env factory with rl‑games
    vecenv.register("RLGPU", lambda n, c, **k: RLGPUEnv(n, c, **k))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": create_env_thunk})

    # ═════════════════════ Runner setup ═════════════════════
    def build_runner(observer):
        runner = Runner(observer)
        runner.algo_factory.register_builder("amp_continuous", lambda **k: amp_continuous.AMPAgent(**k))
        runner.player_factory.register_builder("amp_continuous", lambda **k: amp_players.AMPPlayerContinuous(**k))
        model_builder.register_model("continuous_amp", lambda n, **k: amp_models.ModelAMPContinuous(n))
        model_builder.register_network("amp", lambda **k: amp_network_builder.AMPBuilder())
        return runner

    prefix = cfg.train.params.config.get("user_prefix", "") + cfg.train.params.config.get("auto_prefix", "") + time_str
    cfg.train.params.config.prefix = prefix
    rlg_cfg = omegaconf_to_dict(cfg.train)
    rlg_cfg["params"]["config"]["prefix"] = prefix

    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_cfg)
    runner.reset()

    runner.run({"train": False, "play": True, "checkpoint": cfg.checkpoint, "sigma": 0})

    # Close writer if it exists
    if cfg.capture_video:
        runner.player.env._video_writer.close()

if __name__ == "__main__":
    launch_rlg_hydra()