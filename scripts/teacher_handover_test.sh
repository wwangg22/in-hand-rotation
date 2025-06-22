GPUS=$1
CKPT=/home/william/Desktop/USC/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.0_2025-06-08_19-28-12-3278/nn/last_baoding_ep_5300_rew_1268.9863.pth
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/test.py headless=False \
task.env.objSet=custom task=AllegroArmMOAR task.env.axis=custom \
task.env.numEnvs=1 test=True \
task.env.observationType=full_stack task.env.legacy_obs=True \
task.env.ablation_mode=none experiment=custom \
 wandb_activate=False \
${EXTRA_ARGS}