#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -A C3SE2019-1-14 # group to which you belong
#SBATCH -p vera  # partition (queue)
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=gpu:1
# modules already loaded
# jupyter notebook --no-browser  --ip=0.0.0.0 --port 8888
source $HOME/loadenv_gpu.sh
cd /c3se/users/zrimec/Vera/projects/DeepExpression/2019_3_22
snakemake -j 1 > _run_hyperas_scerevisiae_codons.log
