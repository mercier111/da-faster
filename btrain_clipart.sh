#!/bin/bash
#SBATCH -o job.%j_btrain.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J cycle-bda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 
#SBATCH -w node4 

save_dir="experiments/cycle_DA_Faster_new_gen_6/clipart/model"
dataset="clipart"
net="res101"

python -u  bda_train_net.py --cuda --cag --dataset ${dataset} --net ${net} --save_dir ${save_dir} --max_epochs 8 --lr 0.001 \
                            --nw 5 --lr_decay_step 5  --source_bayes 1 --target_bayes 1 --max_iter 10000 \
                            --bs 1 --lamda 1 --rate 0.1