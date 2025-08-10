GPUS=$1
CKPT=/home/william/Downloads/last_baseline_24acdim_ep_2000_rew_202.7262.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=baseline \
task.env.numEnvs=2 train.params.config.minibatch_size=32 \
train.params.config.central_value_config.minibatch_size=32 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=TEST_test \
wandb_activate=False \
${EXTRA_ARGS}