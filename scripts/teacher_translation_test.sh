GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/translation-test/S1.0_C0.0_M0.02025-07-21_17-56-15-43458/nn/translation-test.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=translation \
task.env.numEnvs=2 train.params.config.minibatch_size=1 \
train.params.config.central_value_config.minibatch_size=1 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
test=True \
checkpoint=${CKPT} \
task.env.ablation_mode=no-pc experiment=z-axis-working \
wandb_activate=False \
${EXTRA_ARGS}