GPUS=$1
CKPT=/home/william/Downloads/last_baseline_ep_17000_rew_211.13008.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=baseline \
task.env.numEnvs=4096 train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=baseline_24acdim \
wandb_activate=True \
${EXTRA_ARGS}