#!/bin/bash

echo "Activate <conda_env> conda environment ..."
export PYTHON_EXEC="$HOME/miniconda3/envs/<conda_env>/bin/python"

echo "Start experiments via slurm ..."
$PYTHON_EXEC -c "import exputils
exputils.start_experiments(directory='./experiments/',
        start_scripts='run_experiment.py',
        start_command='$PYTHON_EXEC {}',
        is_parallel=True,
        is_chdir=True,
        verbose=True,
        post_start_wait_time=0.1)"
echo "Finished"


