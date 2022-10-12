#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=work_dirs/srun_output/nrw_r50_psp_%j.out
#SBATCH --error=work_dirs/srun_output/nrw_r50_psp_%j.err
#SBATCH --time=06:00:00
#SBATCH --job-name=nrw_r50_psp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

sh tools/dist_train.sh configs/pspnet/pspnet_r50-d8_1024x1024_40k_geonrw.py 4 --work-dir work_dirs/pspnet_r50-d8_1024x1024_40k_geonrw --auto-resume
