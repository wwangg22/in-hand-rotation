GPUS=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=False \
task.env.objSet=C task=AllegroArmMOAR task.env.axis=x \
task.env.numEnvs=2 train.params.config.minibatch_size=16 \
train.params.config.central_value_config.minibatch_size=16 \
task.env.observationType=full_stack_pointcloud task.env.legacy_obs=True \
task.env.ablation_mode=no-pc experiment=x-axis \
train.params.config.user_prefix=x-axis wandb_activate=False \
${EXTRA_ARGS}