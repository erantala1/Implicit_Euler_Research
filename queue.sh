#!/bin/bash  
### Job Name
#PBS -N FNO_Implicit_Euler_find_grads_1k
#PBS -e FNO_Implicit_Euler_find_grads_1k.err
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

 python -u eigen_analysis.py /glade/derecho/scratch/erantala/project_runs/outputs/KS_pred_Implicit_Euler_step_FNO_jacs_for_lead_100