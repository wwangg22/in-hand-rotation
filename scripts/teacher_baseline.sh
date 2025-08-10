GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/baseline_gated_w_trans/S1.0_C0.0_M0.02025-08-05_17-25-01-3278/nn/last_baseline_gated_w_trans_ep_7000_rew_78.724396.pth
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=working task=AllegroArmMOAR task.env.axis=baseline \
task.env.numEnvs=2048 train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
task.env.observationType=full_stack_obj_sem task.env.legacy_obs=False \
task.env.ablation_mode=no-pc experiment=baseline_NO_HIGH_LEVEL \
wandb_activate=True \
${EXTRA_ARGS}