GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/z-axis-cup/S1.0_C0.0_M0.02025-06-18_22-38-07-3278/nn/z-axis-cup.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=1024 train.params.config.minibatch_size=2048 \
train.params.config.central_value_config.minibatch_size=2048 \
task.env.observationType=full_stack task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=z-axis-working \
wandb_activate=True \
${EXTRA_ARGS}