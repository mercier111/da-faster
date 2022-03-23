#!/bin/bash
#SBATCH -o job.%j_test.out
#SBATCH -p compute
#SBATCH --qos=low
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=827174975@qq.com 


#python -u test_ssdd.py 
python -u test_levir.py 