# Robot Synesthesia Codebase

This is a work-in-progress repository for training both a LEAP hand / Allegro hand to perform in-hand manipulation tasks

This repo was originally forked from:

**[Robot Synesthesia: In-Hand Manipulation with Visuotactile Sensing](https://yingyuan0414.github.io/visuotactile/)**


## Preparation

We suggest using conda environment with python 3.8. Install **Isaac Gym Preview 4 release** on your laptop (should have a GPU) and server (follow the instructions on the NVIDIA's website, you need to register an account). Other required packages include **pytorch3d**, **hydra-core**, **ray**, **tensorboard**, **wandb**, etc. You may install them via **pip**.

For detailed instruction, see [install.md](install.md).

## Launch Training
### Teacher Policy Training
(1) To train a general rotation policy on Leap hand:
```
scripts/teacher_working.sh 0 task=LEAPArmMOAR task.name=LEAPArmMOAR
```
to train it on the AllegroHand, run:
```
scripts/teacher_working.sh 0 task=AllegroArmMOAR
```

To edit the object set that the policy is trained on, edit the isaacgymenvs/tasks/allegro_arm_morb_axis.py or isaacgymenvs/tasks/leap_arm_morb_axis.py

## Training Adaptation Policy + Object Semantics Policy

The branch transformers implements training scripts for both the Adaptation policy and object semantics policy
