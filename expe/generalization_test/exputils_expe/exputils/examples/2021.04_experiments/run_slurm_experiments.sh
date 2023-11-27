#!/bin/bash

echo "Activate <conda_env> conda environment ..."
export PYTHON_EXEC="$WORK/miniconda3/envs/<conda_env>/bin/python"

echo "Start experiments via slurm ..."
$PYTHON_EXEC -c "import exputils
exputils.start_slurm_experiments(directory='./experiments/', 
				 start_scripts='run_experiment.slurm', 
				 is_parallel=True, 
				 verbose=True,
				 post_start_wait_time=0.5)"

echo "Finished"

