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


def avg_speed_test(observations,SX,SY):
    
    x = torch.arange(SX)
    y = torch.arange(SY)
    yy = y.repeat(SX, 1)
    xx = x.view(-1, 1).repeat(1, SY)
    X = (xx ).double()
    Y = (yy ).double()
    
    
    t_1=150
    filtered_im =observations[t_1:300,:,:,0]
    
    X=X.unsqueeze(0)
    Y=Y.unsqueeze(0)
    mu_0 = filtered_im.sum(axis=(1,2))
    p_x = ((X * filtered_im).sum(axis=(1,2)) / (mu_0+1e-10))
    p_y = ((Y * filtered_im).sum(axis=(1,2)) / (mu_0+1e-10))
    dx=p_x[1:]-p_x[:-1]
    dy=p_y[1:]-p_y[:-1]
    dx=dx- (dx>40)*256 +(dx<-40)*256 
    dy=dx- (dy>40)*256 +(dy<-40)*256
    speeds= torch.sqrt(  (dx)**2  +  (dy)**2  )
     
    filter_d=((observations[t_1+1:300,:,:,0].sum(axis=(1,2))>10 ) * (observations[t_1+1:300,:,:,0].sum(axis=(1,2))<observations[3,:,:,0].sum(axis=(0,1))*4))
    speeds=speeds *filter_d
 
    avg_speed = speeds.sum()/(speeds.shape[0])
    
    return avg_speed
    
    
    
    
    
    
    
	

def try_robustness_obstacles(system,crea,obstacles_pool):
  system.reset(initialization_parameters=crea['initialization'],
                update_rule_parameters=crea['update_rule'])
  fail=[]
  
  ###   test surival random obstacles 
  perf=0
  n=obstacles_pool.shape[0]
  avg_speed=0
  n=obstacles_pool.shape[0]
  system.config.final_step=301
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
    
    avg_speed=avg_speed+avg_speed_test(observations,system.config.SX,system.config.SY)/n
 
    

  
  return perf,fail,avg_speed



def try_robustness(system,crea,random_obstacles_pool,hard_env_pool):
  system.reset(initialization_parameters=crea['initialization'],
                update_rule_parameters=crea['update_rule'])
  performances=[]
  fail=[]
  speed=0
  #### test survival without obstacles
  number_timesteps=301
  system.config.final_step=number_timesteps
  system.random_obstacle(0)
  system.generate_init_state()
  with torch.no_grad():
    observations=system.run()
  observations= observations.states
  speed=avg_speed_test(observations,system.config.SX,system.config.SY)
  print("speed=" +str(speed))
  

  
  ###   test surival random obstacles 
  perf=0
  n=len(random_obstacles_pool)
  perf,fails_env,avg_speed_rd=try_robustness_obstacles(system,crea,random_obstacles_pool)
  performances.append(perf)
  #fail.extend(fails_env)

  ### test survival hard env 
  perf=0
  n=len(hard_env_pool)
  perf,fails_env,avg_speed_hd=try_robustness_obstacles(system,crea,hard_env_pool)
  performances.append(perf)
  #fail.extend(fails_env)
  
  
  return performances,fail,avg_speed_rd,avg_speed_hd
  
  



  

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
  system.random_obstacle_bis(15)
  random_obstacles_pool[i]=system.init_wall*1.0

hard_env_pool=torch.load("../../../hand_obstacles.pickle")





list_speeds_rd=[]
list_avg_speed_rd=[]
list_speeds_hd=[]
list_avg_speed_hd=[]

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
    performances,_,avg_speed_rd,avg_speed_hd=try_robustness(system,policy_parameters,random_obstacles_pool,hard_env_pool)
    print(i,performances,avg_speed_rd,avg_speed_hd)
    list_avg_speed_rd.append(avg_speed_rd)
    list_avg_speed_hd.append(avg_speed_hd)
  else:
    list_avg_speed_rd.append(0)
    list_avg_speed_hd.append(0)

list_avg_speed_rd=np.array(list_avg_speed_rd)
list_avg_speed_hd=np.array(list_avg_speed_hd)


np.save("exp"+str(expe)+"_rep"+str(seed)+"_speed_rd.npy",list_avg_speed_rd)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_speed_hd.npy",list_avg_speed_hd)


print("max speed random "+ str(np.max(list_avg_speed_rd)))
print("max speed hard "+ str(np.max(list_avg_speed_hd)))

print("finished")

