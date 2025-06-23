#!/bin/bash  
### Job Name
#PBS -N FNO_Euler_find_grads_1k
#PBS -e FNO_Euler_find_grads_1k.err
### Charging account
#PBS -A UCSC0009
#PBS -l walltime=12:00:00
#PBS -q main
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=100GB
#PBS -l gpu_type=a100

module load conda
#conda activate /glade/work/cainslie/conda-envs/cainslie_env
conda activate /glade/work/erantala/conda-envs/erantala_env
cd /glade/derecho/scratch/cainslie/conrad_net_stability/Conrad_Research_4_Ashesh/

python -u -m eval_w_jacs.py