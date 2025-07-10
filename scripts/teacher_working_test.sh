GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/z-axis-working/S1.0_C0.0_M0.02025-07-10_02-30-06-3278/nn/last_z-axis-working_ep_11100_rew_99.671265.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=19 train.params.config.minibatch_size=1 \
train.params.config.central_value_config.minibatch_size=1 \
task.env.observationType=full_stack task.env.legacy_obs=False \
test=True \
checkpoint=${CKPT} \
task.env.ablation_mode=no-pc experiment=z-axis-working \
wandb_activate=False \
${EXTRA_ARGS}