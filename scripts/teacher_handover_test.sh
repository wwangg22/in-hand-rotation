GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/custom-powerdrill/S1.0_C0.0_M0.02025-07-08_16-47-25-3278/nn/custom-powerdrill.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/test2.py headless=False \
task.env.objSet=custom task=AllegroArmMOAR task.env.axis=custom \
task.env.numEnvs=1 test=True \
task.env.observationType=full_stack task.env.legacy_obs=True \
task.env.ablation_mode=none experiment=custom-powerdrill \
checkpoint=${CKPT} \
wandb_activate=False \
${EXTRA_ARGS}