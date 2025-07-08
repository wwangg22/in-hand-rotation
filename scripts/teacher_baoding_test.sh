GPUS=$1
CKPT=/home/william/Downloads/last_baoding_ep_500_rew_1228.5563.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/test2.py headless=False \
task.env.objSet=ball task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=1 test=True \
task.env.observationType=full_stack_baoding task.env.legacy_obs=True \
task.env.ablation_mode=none experiment=baoding \
checkpoint=${CKPT} \
wandb_activate=False \
${EXTRA_ARGS}