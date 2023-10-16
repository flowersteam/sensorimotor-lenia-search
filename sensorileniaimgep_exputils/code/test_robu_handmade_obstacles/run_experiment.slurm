#!/bin/bash

#SBATCH -A imi@v100
#SBATCH --time=19:30:00
#SBATCH --input=none
#SBATCH --mail-user=gautier.hamon@inria.fr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err


STATUSFILE=run_experiment.slurm.status




echo "Run the experiment ..."
STATE='Running'

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

python run_experiment.py
RETURN_CODE=$?

echo "Write status file ..."
if [ $RETURN_CODE == 0 ] 
then
	STATE='Finished'
else
	STATE='Error'
fi

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >> $STATUSFILE

echo "Finished."




