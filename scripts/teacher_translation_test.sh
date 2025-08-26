GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/translation-leaphand/S1.0_C0.0_M0.02025-08-10_00-59-42-3278/nn/last_translation-leaphand_ep_2000_rew_103.38379.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=LEAPArmMOAR task.name=LEAPArmMOAR task.env.axis=translation \
task.env.numEnvs=16 train.params.config.minibatch_size=1 \
train.params.config.central_value_config.minibatch_size=1 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
test=True \
checkpoint=${CKPT} \
task.env.ablation_mode=no-pc experiment=z-axis-working \
wandb_activate=False \
${EXTRA_ARGS}