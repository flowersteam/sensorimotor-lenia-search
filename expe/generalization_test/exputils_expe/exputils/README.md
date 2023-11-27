
# Introduction

Experiment Utilities (exputils) contains various tool for the execution of experimental campaigns. 


# Acknowledgements

This package was developped by [Chris Reinke](http:www.scirei.net): <chris.reinke@inria.fr>


The original repository can be found in the flowersteam repository [exputils](https://github.com/flowersteam/automated_discovery_of_lenia_patterns/tree/master/autodisc/exputils).

# Exputils: Install and Run

## Step 1: Installation
1. Clone exputils into your package folder: 
`cd <path_to_packages_folder>`
`git clone git@github.com:mayalenE/exputils.git`  
2. Install the dependencies in your python environment:
`conda install -c conda-forge odfpy`
`conda install -c anaconda numpy`
4. [Recommended to call exputils from any location on your computer]: Add the path to exputils package in your python environment
`echo <path_to_packages_folder> "$HOME/miniconda3/envs/<conda_env>/lib/python3.6/site-packages/my_packages.pth"`


## Step 2: Prepare the experiment folder structures
Experiments are stored in a specific folder structure which allows to save and load experimental data in a structured manner.
Please note that  it represents a default structure which can be adapted if required.
Elements in brackets (\<custom name>\) can have custom names.   
Folder structure:

        <experimental campaign>/  
        ├── analyze                                 # Scripts such as Jupyter notebooks to analyze the different experiments in this experimental campaign.  
        ├── experiment_configurations.ods           # ODS file that contains the configuration parameters of the different experiments in this campaign.  
        ├── code                                    # Holds code templates of the experiments.  
        │   ├── <repetition code>                   # Code templates that are used under the repetition folders of the experiments. These contain the experimental code that should be run.  
        │   ├── <experiment code>                   # Code templates that are used under the experiment folder of the experiment. These contain usually code to compute statistics over all repetitions of an experiment.  
        ├── generate_code.sh                        # Script file that generates the experimental code under the experiments folder using the configuration in the experiment_configurations.ods file and the code under the code folder.          
        ├── experiments folder                      # Contains generated code for experiments and the collected experimental data.
        │   ├── experiment_{id}
        |   │    ├── repetition_{id}
        │   │    │    ├── data                      # Experimental data for the single repetitions, such as logs.
        │   │    │    └── <code scripts>                    # Generated code and resource files.
        |   │    ├── data                           # Experimental data for the whole experiment, e.g. statistics that are calculated over all repetitions.   
        |   │    └── <code scripts>                         # Generated code and resource files.  
        └── <run scripts>.sh                        # Various shell scripts to run experiments and calculate statistics locally or on clusters.


## Step 3: Run the experiments on clusters
### Installation of conda env on clusters

1. install Miniconda on the cluster:
```bash
  cd /tmp
  wget <https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh>
  chmod +x Miniconda-latest-Linux-x86_64.sh
  ./Miniconda-latest-Linux-x86_64.sh 
  # follow the installation instruction (path for the installation, conda init: yes)
  source ~/.bashrc # activate the installation
```
2. Setup the conda environnement
3. Create folders <DESTINATION_CODE> and <DESTINATION_EXPERIMENT> on clusters and modify accordingly <run_scripts>.sh 


### Useful commands on the clusters
```
---------
Slurm

- See running jobs: 
	squeue -u <USERNAME>

- Detect status of experiments and calculation of statistics:
	for f in $(find . -name "run_experiment.slurm.status"); do STATUS=$(tail -1 $f); echo $STATUS - $f ;done
	for f in $(find . -name "run_calc_statistics_per_experiment.slurm.status"); do STATUS=$(tail -1 $f); echo $STATUS - $f ;done
	for f in $(find . -name "run_calc_statistics_per_repetition.slurm.status"); do STATUS=$(tail -1 $f); echo $STATUS - $f ;done

- Deleting specific files:
	find -name <FILENAME> -delete
```

---
