#!/bin/bash
#SBATCH -o job.%j_train.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J ssdd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 

save_dir="experiments/DA_Faster/SSDD/model"
dataset="SSDD"
net="res101"

python -u  da_train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} \
                           --max_epochs 30  --lr_decay_step 10 --lr 0.001 --max_iter 2000