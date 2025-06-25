module load conda
conda activate /glade/work/erantala/conda-envs/jacobian_env

# sanity-check
which python
python -c "import numpy, torch, sys; print(sys.executable)"