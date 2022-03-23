#!/bin/bash
#SBATCH -o job.%j_test.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J cycle_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 



python -u train/cycle_faster/test_ssdd.py