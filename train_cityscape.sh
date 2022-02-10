#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J FirstJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 
#SBATCH -w node4
save_dir="/data/experiments/DA_Faster_ICR_CCR/cityscape/model"
dataset="cityscape"
net="vgg16"
pretrained_path="/data/pretrained_model/vgg16_caffe.pth"

python da_train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path}  --max_epochs 12
