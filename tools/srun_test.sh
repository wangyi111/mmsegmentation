#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=work_dirs/srun_output/nrw_test_%j.out
#SBATCH --error=work_dirs/srun_output/nrw_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --job-name=nrw_test
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

sh tools/dist_test.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_1024x1024_40k_geonrw.py work_dirs/deeplabv3plus_r50-d8_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/deeplabv3plus_r50-d8_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/pspnet/pspnet_r50-d8_1024x1024_40k_geonrw.py work_dirs/pspnet_r50-d8_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/pspnet_r50-d8_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/upernet/upernet_r50_1024x1024_40k_geonrw.py work_dirs/upernet_r50_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/upernet_r50_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/convnext/upernet_convnext_tiny_fp16_1024x1024_40k_geonrw.py work_dirs/upernet_convnext_tiny_fp16_1024x1024_40k_geonrw/iter_36000.pth 4 --work-dir work_dirs/upernet_convnext_tiny_fp16_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/vit/upernet_vit-s16_mln_1024x1024_40k_geonrw.py work_dirs/upernet_vit-s16_mln_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/upernet_vit-s16_mln_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/swin/upernet_swin_tiny_patch4_window7_1024x1024_40k_geonrw_pretrain_224x224_1K.py work_dirs/upernet_swin_tiny_patch4_window7_1024x1024_40k_geonrw_old/iter_40000.pth 4 --work-dir work_dirs/upernet_swin_tiny_patch4_window7_1024x1024_40k_geonrw_old --eval mIoU &&
sh tools/dist_test.sh configs/unet/fcn_unet_s5-d16_1024x1024_40k_geonrw.py work_dirs/fcn_unet_s5-d16_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/fcn_unet_s5-d16_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/segformer/segformer_mit-b3_1024x1024_40k_geonrw.py work_dirs/segformer_mit-b3_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/segformer_mit-b3_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/deeplabv3plus/deeplabv3plus_convnext-tiny_1024x1024_40k_geonrw.py work_dirs/deeplabv3plus_convnext-tiny_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/deeplabv3plus_convnext-tiny_1024x1024_40k_geonrw --eval mIoU &&
sh tools/dist_test.sh configs/deeplabv3plus/deeplabv3plus_swin-tiny_1024x1024_40k_geonrw.py work_dirs/deeplabv3plus_swin-tiny_1024x1024_40k_geonrw/iter_40000.pth 4 --work-dir work_dirs/deeplabv3plus_swin-tiny_1024x1024_40k_geonrw --eval mIoU &&

echo "Finish!"