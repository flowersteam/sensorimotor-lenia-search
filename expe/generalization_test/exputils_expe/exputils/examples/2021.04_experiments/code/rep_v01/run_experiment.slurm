#!/bin/bash

#SBATCH -A imi@cpu
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=00:01:00
#SBATCH --input=none
#SBATCH --mail-user=mayalen.etcheverry@inria.fr
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err


STATUSFILE=run_experiment.slurm.status


echo "Activate <conda_env> conda environment ..."
export PYTHON_EXEC="$WORK/miniconda3/envs/<conda_env>/bin/python"


echo "Run the experiment ..."
STATE='Running'

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

$PYTHON_EXEC run_experiment.py
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




