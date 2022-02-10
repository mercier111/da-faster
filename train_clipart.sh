#!/bin/bash
#SBATCH -o job.%j_train.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J clipart
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 

save_dir="experiments/DA_Faster_bayes/clipart/model"
dataset="clipart"
net="res101"

python -u  da_train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} \
                           --max_epochs 8  --lr_decay_step 5 --lr 0.001 --bayes
