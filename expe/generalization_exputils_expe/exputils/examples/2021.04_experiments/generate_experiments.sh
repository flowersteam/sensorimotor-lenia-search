#!/bin/bash 

echo "Activate <conda_env> conda environment ..."
source ~/miniconda3/bin/activate <conda_env>

echo "Generate experiments ..."
python -c "import exputils
exputils.generate_experiment_files('experiment_configurations.ods', directory='./experiments/')"

echo "Finished."

$SHELL
