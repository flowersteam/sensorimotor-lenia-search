# Discovering Sensorimotor Agency in Cellular Automata using Diversity Search 

Code to reproduce the experiments from the paper : Discovering Sensorimotor Agency in Cellular Automata using Diversity Search 
Gautier Hamon¹, Mayalen Etcheverry¹, Bert Wang-Chak Chan², Clément Moulin-Frier¹, and Pierre-Yves Oudeyer¹ 
1. Inria Bordeaux (Flowers team), 2. Google Deepmind


[Companion website for videos and demo](https://developmentalsystems.org/sensorimotor-lenia-companion/)

[Colab for the main method (IMGEP search)] (https://colab.research.google.com/drive/11mYwphZ8I4aur8KuHRR1HEg6ST5TI0RW?usp=sharing)

Contact us for questions or comments: gautier.hamon@inria. fr



## Instructions


### Installation 

Install sensorimotor_leniasearch 

```
pip install sensorimotor_leniasearch/.
```


### Running experiments 

Experiments are under the expe folder 

For example to run one seed of imgep search run :

```
python expe/imgep/run_experiment.py
```
For each expe you can change expe_config.py to change seed, path etc. 


After running a search, you can run prefilter (changing the config for the parameters generated)

```
python expe/prefilter/run_experiment.py
```

Then run the stats on long rollout with 

```
python expe/stats_from_params/run_experiment.py
```

Then on the generated stats run calc_categories.py to run the empirical agency and moving test 
```
python expe/calc_categories.py
```

Then run the empirical base robustness test 
```
python expe/test_robu/run_experiment.py
```


For the generalization tests, we provide in expe/generalization_test all tests performed in the paper (same run python expe/generalization_test/expe_name/run_experiment.py). 
For slurm users, we also provide in expe/generalization_test/exputils_expe code to launch several seeds on all the generalization tests. change experiment_configurations according to your need. Then run generate_experimets.sh to generate the experiment folder and run run_slurm_experiments.ssh to launch all experiments. 
The *exputils* packages used here builds upon [flowersteam's packages](https://github.com/flowersteam/automated_discovery_of_lenia_patterns) developped by [Chris Reinke](https://www.scirei.net/). 



### Parameters from the original Lenia papers (handmade search)

The code to gather the parameters from the original papers is in expe/handmade_crea_json2pickle.py . It gathers parameters from [https://github.com/Chakazul/Lenia/blob/master/Python/animals.json](https://github.com/Chakazul/Lenia/blob/master/Python/animals.json) and [https://github.com/Chakazul/Lenia/tree/master/Python/found](https://github.com/Chakazul/Lenia/tree/master/Python/found)

### Data and reproducing the figures 

Notebooks and data to reproduce the figures of the paper can be found in the data folder 









