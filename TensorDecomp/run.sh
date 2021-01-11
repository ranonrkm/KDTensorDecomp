#!/bin/bash

#SBATCH -J hsd
#SBATCH -o out_hsd
#SBATCH -e err_hsd
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ranajoy@iitkgp.ac.in
#SBATCH --mail-type=ALL

module load compiler/cuda/10.1

source activate /home/16ee35016/anaconda2/envs/HSD
cd $SCRATCH
cd BTP/KDTensorDecomp/TensorDecomp

export DATA_PATH='/scratch/16ee35016/data'

python -u main.py -b 32 --print-freq 50 --pretrained --decompose --teacher --gpu 0 /scratch/16ee35016/data > test_o.out

