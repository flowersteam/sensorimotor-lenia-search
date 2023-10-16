#!/bin/bash



echo "Start experiments via slurm ..."
python -c "import exputils
exputils.start_slurm_experiments(directory='./experiments/experiment_001004' ,
				 start_scripts='run_experiment.slurm', 
				 is_parallel=True, 
				 verbose=True,
				 post_start_wait_time=1.)"
echo "Finished"

