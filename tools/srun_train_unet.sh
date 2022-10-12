#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=work_dirs/srun_output/nrw_unet_fcn_%j.out
#SBATCH --error=work_dirs/srun_output/nrw_unet_fcn_%j.err
#SBATCH --time=04:00:00
#SBATCH --job-name=nrw_unet
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

sh tools/dist_train.sh configs/unet/fcn_unet_s5-d16_1024x1024_40k_geonrw.py 4 --work-dir work_dirs/fcn_unet_s5-d16_1024x1024_40k_geonrw --auto-resume
