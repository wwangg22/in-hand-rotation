GPUS=$1
CKPT=/home/william/Downloads/last_z-axis-working-objsem-w-obj-enc-8-dim_ep_6000_rew_1713.696.pth
# LOC=/home/william/Downloads
# NAME=last_baseline_6d_quat_goal_6d_quat_pose_ep_3000_rew_436.86972
LOC=/home/william/Desktop/USC/in-hand-rotation/runs/baseline_NO_HIGH_LEVEL/S1.0_C0.0_M0.02025-08-09_04-59-05-3278/nn
NAME=last_baseline_NO_HIGH_LEVEL_ep_12000_rew_114.52683
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/test.py headless=True \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=baseline \
task.env.numEnvs=128 train.params.config.minibatch_size=2 \
train.params.config.central_value_config.minibatch_size=2 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
test=True distill.bc_training=warmup \
distill.teacher_logdir=${LOC} \
distill.teacher_resume=${NAME} \
distill.high_level_planner=False \
task.env.ablation_mode=no-pc experiment=z-axis-working \
wandb_activate=False \
${EXTRA_ARGS}