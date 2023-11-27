#!/bin/bash

#SBATCH -A imi@v100
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --input=none
#SBATCH --mail-user=mayalen.etcheverry@inria.fr
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




