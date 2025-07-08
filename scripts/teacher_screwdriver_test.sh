GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/z-axis-cup/S1.0_C0.0_M0.02025-06-18_22-52-58-3278/nn/last_z-axis-cup_ep_3600_rew_556.7728.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/test2.py headless=False \
task.env.objSet=custom task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=1 test=True \
task.env.observationType=full_stack task.env.legacy_obs=True \
task.env.ablation_mode=none experiment=z-screwdriver \
wandb_activate=False \
${EXTRA_ARGS}