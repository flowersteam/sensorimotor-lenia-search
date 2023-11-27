#!/bin/bash

SOURCECODE=/home/mayalen/code/my_packages
DESTINATIONCODE=<id_jeanzay>@jeanzay:/gpfswork/rech/zaj/<id_jeanzay>/code/
DESTINATIONEXPERIMENT=<id_jeanzay>@jeanzay:/gpfsscratch/rech/zaj/<id_jeanzay>/code/2021.04_experiments

echo Transfer code ...
rsync -azvh -e "ssh -i $HOME/.ssh/id_rsa_jeanzay" --exclude "*idea/" --exclude "*git/" --exclude "*__pycache__/" --exclude "*.pyc" $SOURCECODE $DESTINATIONCODE
rsync -azvh -e "ssh -i $HOME/.ssh/id_rsa_jeanzay" run_slurm_experiments.sh $DESTINATIONEXPERIMENT

echo Transfer experiments ...
rsync -azvh  --exclude "*__pycache__/" --exclude "*idea/" --exclude "*data/" -e "ssh -i $HOME/.ssh/id_rsa_jeanzay" experiments $DESTINATIONEXPERIMENT

$SHELL
