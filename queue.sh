#!/bin/bash  
### Job Name
#PBS -N FNO_Implicit_Euler
#PBS -e FNO_Implicit_Euler.err
### Charging account
#PBS -A UCSC0009
#PBS -l walltime=12:00:00
#PBS -q main
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=100GB
#PBS -l gpu_type=a100

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /glade/work/erantala/conda-envs/jacobian_env

cd /glade/derecho/scratch/erantala/project_runs/code

python -u nn_train_multistep.py