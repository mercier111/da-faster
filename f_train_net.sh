#!/bin/bash
#SBATCH -o job.%j_btrain.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J f-faster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 
#SBATCH -w node4 

save_dir="experiments/f_Faster/clipart/model"
dataset="clipart"
net="res101"

python -u  f_train_net.py --cuda --cag --dataset ${dataset} --net ${net} --save_dir ${save_dir} --max_epochs 12 --lr 0.001 \
                            --nw 5 --lr_decay_step 5 --max_iter 10000 \
                            --bs 1 --lamda  0.1