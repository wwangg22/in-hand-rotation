GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/baseline_LEAPHAND_rel_quat/S1.0_C0.0_M0.02025-08-21_00-53-48-3278/nn/last_baseline_LEAPHAND_rel_quat_ep_4000_rew_382.38263.pth
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=working task=LEAPArmMOAR task.name=LEAPArmMOAR task.env.axis=baseline \
task.env.numEnvs=16 train.params.config.minibatch_size=32 \
train.params.config.central_value_config.minibatch_size=32 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=TEST_test \
checkpoint=${CKPT} \
wandb_activate=False \
${EXTRA_ARGS}