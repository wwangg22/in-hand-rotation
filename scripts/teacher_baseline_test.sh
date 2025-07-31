GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/TEST/S1.0_C0.0_M0.02025-07-30_21-25-11-28906/nn/TEST.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=baseline \
task.env.numEnvs=2048 train.params.config.minibatch_size=8192 \
train.params.config.central_value_config.minibatch_size=8192 \
checkpoint=${CKPT} \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=TEST \
wandb_activate=True \
${EXTRA_ARGS}