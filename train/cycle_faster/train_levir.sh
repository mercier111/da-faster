#!/bin/bash
#SBATCH -o job.%j_train.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J img+norm_da_ssdd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 

save_dir="experiments/Cylce_Faster_norm+img_da/LEVIR/model"
dataset="LEVIR"
net="res101"

python -u  train/cycle_faster/train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} \
                           --max_epochs 12  --lr_decay_step 8 --lr 0.001 --max_iter 1500 --cag \
                           --use_norm_da --use_img_da

                           # --chaos
                           # --attention
                           #--use_img_da 
                           #--use_norm_da 
                           #--use_detect_da 
                           #--use_sk_da 
                           #--use_instance_da