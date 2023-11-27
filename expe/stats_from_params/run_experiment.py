from experiment_config import *
from stats_utils import *
import torch
import os
from addict import Dict
from sensorimotorleniasearch.systems import Lenia_C
from sensorimotorleniasearch.systems import Lenia_C_move
import numpy as np
import torch
from sensorimotorleniasearch.systems.lenia import LeniaInitializationSpace
import cv2


torch.manual_seed(seed)
np.random.seed(seed)





  

lenia_config = Lenia_C.default_config()
lenia_config.SX = 256
lenia_config.SY = 256
lenia_config.final_step = 2000
lenia_config.version = 'pytorch_fft'
lenia_config.nb_kernels=10

initialization_space_config = Dict()
initialization_space = LeniaInitializationSpace(config=initialization_space_config)

system = Lenia_C(initialization_space=initialization_space, config=lenia_config, device='cuda')
SX=system.config.SX

count=0

path=path_parameters

for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file)) and file.startswith('seed'+str(seed)):


        if(count%100==0):
            print("count at "+str(count))
        crea_params=torch.load(os.path.join(path,file))
        policy_parameters = crea_params["policy_parameters"]
        system.reset(initialization_parameters=policy_parameters['initialization'],
        update_rule_parameters=policy_parameters['update_rule'])



        #run the system
        with torch.no_grad():
            system.config.final_step = 2000
            system.random_obstacle(0)
            system.generate_init_state_bis()
            observations = system.run()
            observations=observations.states


            #uncomment if you want to save the observations
            #torch.save(observations,os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/resources/"+type_expe+"_exploration/observations/observations_"+file)
            
            statistics = calc_statistics(observations[:,:,:,0].numpy(), crea_params)
                
            torch.save(statistics,path_outdata+"stats/stats_"+file)
            
            print(file,statistics['connected_components_nr_objects'])
            
            
            count=count+1
        
        

print("total count" +str(count))
print("finished")

