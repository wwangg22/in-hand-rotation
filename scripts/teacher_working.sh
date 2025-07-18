GPUS=$1
CKPT=/home/william/in-hand-manipulation/in-hand-rotation/runs/z-axis-working-objsem-w-rot-32dim-z-reward/S1.0_C0.0_M0.02025-07-16_12-09-38-84114/nn/last_z-axis-working-objsem-w-rot-32dim-z-reward_ep_10000_rew_157.7412.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=4096 train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=z-axis-working-objsem-w-rot-working-obj-only \
wandb_activate=True \
${EXTRA_ARGS}