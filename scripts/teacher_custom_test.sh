GPUS=$1
CKPT=/home/william/Downloads/last_z-axis-knife_ep_7000_rew_389.66296.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=custom task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=6 test=True \
task.env.observationType=full_stack task.env.legacy_obs=True \
task.env.ablation_mode=no-pc experiment=z-axix-knife \
checkpoint=${CKPT} \
wandb_activate=False \
${EXTRA_ARGS}