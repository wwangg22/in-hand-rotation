GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/translation-leaphand/S1.0_C0.0_M0.02025-08-09_23-54-52-3278/nn/translation-leaphand.pth
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=working task=LEAPArmMOAR task.name=LEAPArmMOAR task.env.axis=translation \
task.env.numEnvs=2048 train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=translation-leaphand \
checkpoint=${CKPT} \
wandb_activate=True \
${EXTRA_ARGS}