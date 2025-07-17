GPUS=$1
CKPT=/home/william/Downloads/last_z-axis-working-objsem-w-rot-32dim-new_ep_16000_rew_220.98907.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=5 train.params.config.minibatch_size=1 \
train.params.config.central_value_config.minibatch_size=1 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
test=True \
checkpoint=${CKPT} \
task.env.ablation_mode=no-pc experiment=z-axis-working \
wandb_activate=False \
${EXTRA_ARGS}