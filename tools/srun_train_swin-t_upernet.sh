#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=work_dirs/srun_output/nrw_upernet_swint_%j.out
#SBATCH --error=work_dirs/srun_output/nrw_upernet_swint_%j.err
#SBATCH --time=05:00:00
#SBATCH --job-name=nrw_upsw
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

sh tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_1024x1024_40k_geonrw_pretrain_224x224_1K.py 4 --work-dir work_dirs/upernet_swin_tiny_patch4_window7_1024x1024_40k_geonrw_pretrain_224x224_1K --auto-resume
