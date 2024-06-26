from experiment_config import *
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
lenia_config.final_step = 500
lenia_config.version = 'pytorch_fft'
lenia_config.nb_kernels=10

initialization_space_config = Dict()
initialization_space = LeniaInitializationSpace(config=initialization_space_config)

system = Lenia_C(initialization_space=initialization_space, config=lenia_config, device='cuda')









count=0
for i in range(30000):
    if(i%100==0):
        print("count at "+str(i)+" trials = " +str(count))
 
    policy_parameters = Dict.fromkeys(['initialization', 'update_rule']) 
    policy_parameters['initialization'] = system.initialization_space.sample()
    policy_parameters['update_rule'] = system.update_rule_space.sample()
    system.reset(initialization_parameters=policy_parameters['initialization'],
    update_rule_parameters=policy_parameters['update_rule'])
    
    
    #prefilter
    #run the system
    with torch.no_grad():
        system.config.final_step = 500
        system.random_obstacle(0)
        system.generate_init_state()
        observations = system.run()
        observations=observations.states
        
      
    if(observations[-1,:,:,0].sum()<6400 and observations[-1,:,:,0].sum()>0):
        crea_nb="{:05d}".format(count)
        crea={"policy_parameters":policy_parameters}
        torch.save(crea,os.environ["SCRATCH"]+"/sensorimotor_lenia/resources/random_exploration/seed"+str(seed)+"_crea"+crea_nb+".pickle")
        count=count+1
        
        

print("total count" +str(count))
print("finished")

