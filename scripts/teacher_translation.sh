GPUS=$1
CKPT=/home/william/in-hand-manipulation/in-hand-rotation/runs/translation-test/S1.0_C0.0_M0.02025-07-22_18-21-18-45478/nn/last_translation-test_ep_4000_rew_11.345298.pth
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=translation \
task.env.numEnvs=4096 train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=translation-big-fall-pen-cup \
wandb_activate=True \
${EXTRA_ARGS}