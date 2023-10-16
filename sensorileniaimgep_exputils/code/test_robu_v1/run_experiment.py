from experiment_config import *
import torch
import os
from addict import Dict
from sensorimotorleniasearch.systems import Lenia_C
from sensorimotorleniasearch.systems import Lenia_C_move
import numpy as np
import torch
from sensorimotorleniasearch.systems.lenia import LeniaInitializationSpace




sf="../../experiment_00000"+str(expe)+"/repetition_00000"+str(seed)  


def speed_test(observations,SX,SY):
    
    x = torch.arange(SX)
    y = torch.arange(SY)
    yy = y.repeat(SX, 1)
    xx = x.view(-1, 1).repeat(1, SY)
    X = (xx ).double()
    Y = (yy ).double()
    
    
    t_1=25
    filtered_im =observations[t_1,:,:,0]
    mu_0 = filtered_im.sum()
    p_x_1 = ((X * filtered_im).sum() / (mu_0+1e-10))
    p_y_1 = ((Y * filtered_im).sum() / (mu_0+1e-10))
    
    t_2=50
    filtered_im =observations[t_2,:,:,0]

	# recenter
    mu_0 = filtered_im.sum()
    p_x_2 = ((X * filtered_im).sum() / (mu_0+1e-10))
    p_y_2 = ((Y * filtered_im).sum() / (mu_0+1e-10))
    
    return(torch.sqrt((p_x_2-p_x_1)**2+(p_y_2-p_y_1)**2)/(t_2-t_1))
    
    
    
    
    
    
    
	

def try_robustness_obstacles(system,crea,obstacles_pool):
  system.reset(initialization_parameters=crea['initialization'],
                update_rule_parameters=crea['update_rule'])
  fail=[]
  
  ###   test surival random obstacles 
  perf=0
  n=obstacles_pool.shape[0]
  system.config.final_step=500
  for env in obstacles_pool:
    system.reset(initialization_parameters=crea['initialization'],
                update_rule_parameters=crea['update_rule'])
    system.state[0,:,:,1]=env
    with torch.no_grad():
      observations=system.run()
    observations= observations.states

    if ((observations[-1,:,:,0].sum(axis=(0,1))>10 ) and (observations[-1,:,:,0].sum(axis=(0,1))<observations[3,:,:,0].sum(axis=(0,1))*4)):
      perf=perf+1/n
    else:
      fail.append(env)

  
  return perf,fail



def try_robustness(system,crea,random_obstacles_pool,hard_env_pool):
  system.reset(initialization_parameters=crea['initialization'],
                update_rule_parameters=crea['update_rule'])
  performances=[]
  fail=[]
  speed=0
  #### test survival without obstacles
  number_timesteps=1000
  system.config.final_step=number_timesteps
  system.random_obstacle(0)
  system.generate_init_state()
  #death=np.zeros(1000)
  with torch.no_grad():
    observations=system.run()
  observations= observations.states
  speed=speed_test(observations,system.config.SX,system.config.SY)
  
  collapse=(observations[:,:,:,0].sum(axis=(1,2))>10 )
  explode=(observations[:,:,:,0].sum(axis=(1,2))<observations[3,:,:,0].sum(axis=(0,1))*4)
  death=collapse*explode
  perf=death.detach().cpu().numpy().sum()
  performances.append(perf)

  
  ###   test surival random obstacles 
  perf=0
  n=len(random_obstacles_pool)
  perf,fails_env=try_robustness_obstacles(system,crea,random_obstacles_pool)
  performances.append(perf)
  #fail.extend(fails_env)
  

  ### test survival hard env 
  perf=0
  n=len(hard_env_pool)
  perf,fails_env=try_robustness_obstacles(system,crea,hard_env_pool)
  performances.append(perf)
  #fail.extend(fails_env)
  
  return performances,fail,speed
  
  



  

lenia_config = Lenia_C.default_config()
lenia_config.SX = 256
lenia_config.SY = 256
lenia_config.final_step = 50
lenia_config.version = 'pytorch_fft'
lenia_config.nb_kernels=10

initialization_space_config = Dict()
initialization_space = LeniaInitializationSpace(config=initialization_space_config)

system = Lenia_C(initialization_space=initialization_space, config=lenia_config, device='cuda')





lenia_config = Lenia_C_move.default_config()
lenia_config.SX = 256
lenia_config.SY = 256
lenia_config.final_step = 50
lenia_config.version = 'pytorch_fft'
lenia_config.nb_kernels=10

initialization_space_config = Dict()
initialization_space = LeniaInitializationSpace(config=initialization_space_config)

system_move = Lenia_C_move(initialization_space=initialization_space, config=lenia_config, device='cuda')




nb_env=50
random_obstacles_pool=torch.zeros((nb_env,lenia_config.SX,lenia_config.SY))


for i in range(nb_env):
  system.random_obstacle_bis(30)
  random_obstacles_pool[i]=system.init_wall*1.0

hard_env_pool=torch.load("../../../hand_obstacles.pickle")





p1=[]
p2=[]
p3=[]
p4=[]
speeds=[]
for i in range(160):
  nb=i
  if(i%10==0):
    print(i)
  if(nb<10):
    a=torch.load(sf+"/run_000000"+str(nb)+"_data.pickle")
  elif(nb<100):
    a=torch.load(sf+"/run_00000"+str(nb)+"_data.pickle")
  elif(nb<1000):
    a=torch.load(sf+"/run_0000"+str(nb)+"_data.pickle")
  else:
    a=torch.load(sf+"/run_000"+str(nb)+"_data.pickle")
  if(a["reached_goal"][0]<0.1):
    policy_parameters = Dict.fromkeys(['initialization', 'update_rule']) 
    policy_parameters['initialization']=a['policy_parameters']['initialization']
    policy_parameters['update_rule']=a['policy_parameters']['update_rule']
    performances,_,speed=try_robustness(system,policy_parameters,random_obstacles_pool,hard_env_pool)
    perf,_=try_robustness_obstacles(system_move,policy_parameters,random_obstacles_pool[:50])
    print(a["reached_goal"],performances,perf)
    p1.append(performances[0])
    p2.append(performances[1])
    p3.append(performances[2])
    p4.append(perf)
    speeds.append(speed)
  else:
    p1.append(0)
    p2.append(0)
    p3.append(0)
    p4.append(0)
    speeds.append(0)
p1=np.array(p1)
p2=np.array(p2)
p3=np.array(p3)
p4=np.array(p4)
speeds=np.array(speeds)

np.save("exp"+str(expe)+"_rep"+str(seed)+"_Longstable.npy",p1)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_randomObs.npy",p2)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_hardObs.npy",p3)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_moveObs.npy",p4)
np.save("speeds.npy",speeds)


print("max long term "+ str(np.max(p1)))
print("max random "+ str(np.max(p2)))
print("max hard "+ str(np.max(p3)))
print("max mov  "+ str(np.max(p4)))
print("finished")












