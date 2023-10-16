
from experiment_config import *
from sensorimotorleniasearch import ExplorationDB
from sensorimotorleniasearch.systems import Lenia_C
from sensorimotorleniasearch.systems.lenia import LeniaInitializationSpace
from sensorimotorleniasearch.output_representation import LeniaCentroidRepresentation
from sensorimotorleniasearch.explorers import IMGEPExplorer, BoxGoalSpace
from addict import Dict
from copy import deepcopy
import os
import torch
import warnings
import numpy as np

if __name__ == '__main__':
    print("REP v_01 source code")
    print(f'seed(repetition id): {seed}')
    
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Load System: here lenia
    lenia_config = Lenia_C.default_config()
    lenia_config.SX = 256
    lenia_config.SY = 256
    lenia_config.final_step = 50
    lenia_config.version = 'pytorch_fft'
    lenia_config.nb_kernels=10

    initialization_space_config = Dict()




    initialization_space = LeniaInitializationSpace(config=initialization_space_config)

    system = Lenia_C(initialization_space=initialization_space, config=lenia_config, device='cuda')

    # Load ExplorationDB
    db_config = ExplorationDB.default_config()
    db_config.db_directory = '.'
    db_config.save_observations = False
    db_config.keep_saved_runs_in_memory=False
    db_config.load_observations = True
    exploration_db = ExplorationDB(config=db_config)

    # Load Imgep Explorer


    output_representation_config = LeniaCentroidRepresentation.default_config()
    output_representation_config.env_size = (system.config.SX, system.config.SY)
    output_representation_config.device="cpu"
    output_representation = LeniaCentroidRepresentation(config=output_representation_config)
    goal_space = BoxGoalSpace(output_representation)

    ## Load Goal Space Representation

    # goal_space = BoxGoalSpace(output_representation,low=torch.tensor([0,-0.5,-0.5]),high=torch.tensor([2,0.5,0.5]),autoexpand=False)

    ## Load imgep explorer
    explorer_config = IMGEPExplorer.default_config()
    explorer_config.num_of_random_initialization = 40
    explorer_config.reach_goal_optimizer = Dict()
    explorer_config.reach_goal_optimizer.optim_steps = 125
    explorer_config.reach_goal_optimizer.name = "Adam"
    explorer_config.reach_goal_optimizer.initialization_cppn.parameters.lr = 0.8e-2
    explorer_config.reach_goal_optimizer.lenia_step.parameters.lr = 0.8e-3
    explorer = IMGEPExplorer(system, exploration_db, goal_space, config=explorer_config)


    # Run Imgep Explorer
    explorer.run(160,nb_obstacles=0)

    # # save
    # explorer.save('explorer.pickle')
    #
    #restart from checkpoint
    # explorer = IMGEPExplorer.load('explorer.pickle', load_data=False, map_location='cpu')
    # explorer.db = ExplorationDB(config=db_config)
    # explorer.db.load(map_location='cpu')
    # explorer.run(20, continue_existing_run=True)


    goal_lib_copy=explorer.goal_library*1.0
    goal_lib_copy[:,2]=goal_lib_copy[:,2]+1000*(goal_lib_copy[:,1]<-8)+1000*(goal_lib_copy[:,0]>0.12)
    best_achiever=torch.argmin(goal_lib_copy[:,2])
    print("best achiever is run : "+str(best_achiever))
    print("goal achieved by the best achiever :" +str(explorer.goal_library[best_achiever]))
