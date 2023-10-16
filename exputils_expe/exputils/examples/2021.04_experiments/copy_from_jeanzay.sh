#!/bin/bash

SOURCE=<id_jeanzay>@jeanzay:/gpfsscratch/rech/zaj/<id_jeanzay>/code/2021.04_experiments/

for file in $(find . -type d -name "experiment_*")
do
	echo $file
	rsync -azvh --include "*repetition_*/" --include "*data/" --exclude "*.py" --exclude "*__pycache__/" --exclude "*.slurm" --exclude "*.scs" -e "ssh -i $HOME/.ssh/id_rsa_jeanzay" "${SOURCE}${file}/" ${file}
done


$SHELL
