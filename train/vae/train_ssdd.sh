#!/bin/bash
#SBATCH -o job.%j_train.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J detect_vae2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 



save_dir="experiments/detect_vae/LEVIR/model"
dataset="LEVIR"
net="res101"

python -u  train/vae/train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} \
                           --max_epochs 12  --lr_decay_step 8 --lr 0.001 --max_iter 2000 --cag \
                           --lamda 0.01 --lamda2 10 --use_tensor_board
                           
                           #--use_tensor_board
