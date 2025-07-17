GPUS=$1

NAME=last_baoding_ep_5300_rew_1268.9863.pth
LOC=/home/william/Desktop/USC/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02025-06-08_19-28-12-3278/nn
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train_distillation.py headless=True \
distill.teacher_data_dir=demonstration-working \
task.env.legacy_obs=False distill.bc_training=collect \
task.env.objSet=working task.env.is_distillation=True \
train.params.config.user_prefix=bc-working-collect task=AllegroArmMOAR \
task.env.numEnvs=50 train.params.config.minibatch_size=1024 \
experiment=bc-working-collect wandb_activate=False \
task.env.observationType=full_stack_obj_sem \
train.params.config.central_value_config.minibatch_size=1024 \
distill.worker_id=0 distill.ablation_mode=multi-modality-plus \
task.env.ablation_mode=multi-modality-plus \
${EXTRA_ARGS}