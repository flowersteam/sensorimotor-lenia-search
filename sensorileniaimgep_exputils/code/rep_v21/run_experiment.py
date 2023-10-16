from experiment_config import *
import torch
import os
from addict import Dict
from sensorimotorleniasearch.systems import Old_lenia

import numpy as np
import torch
from sensorimotorleniasearch.systems.lenia import LeniaInitializationSpace
import cv2
import json

seed=seed
torch.manual_seed(seed)
np.random.seed(seed)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

sf="../../experiment_00000"+str(expe)+"/repetition_00000"+str(seed)  


   

    
    
def avg_speed_test(observations,SX,SY,t_1=150,t_2=300,dt=10):
    
    x = torch.arange(SX)
    y = torch.arange(SY)
    yy = y.repeat(SX, 1)
    xx = x.view(-1, 1).repeat(1, SY)
    X = (xx ).double()
    Y = (yy ).double()
    
    
    
    filtered_im =observations[t_1:t_2,:,:,0]*1.0
    dr=torch.zeros((t_2-t_1,2))
    mu_0 = filtered_im.sum(axis=(1,2))
    mid=torch.tensor([SX,SY])//2
    for i in range(t_2-t_1-1):
      #plt.imshow(filtered_im[i].cpu())
      #plt.show()
      p_x = ((X * filtered_im[i]).sum(axis=(0,1)) / (mu_0[i]+1e-10))
      p_y = ((Y * filtered_im[i]).sum(axis=(0,1)) / (mu_0[i]+1e-10))
      #print(p_x,p_y)
      #print(p_x)
      #print(p_y)
      mid_b=torch.tensor([p_x,p_y]).int()
      d=-(mid_b-torch.tensor([SX,SY])//2)
      #print(d)
      dr[i+1]=d
      mid=mid_b
      #print((d[0],d[1]))
      #print(filtered_im[i+1].shape)
      
      filtered_im[i+1:]=torch.roll(filtered_im[i+1:],(d[0],d[1]),dims=(1,2))
     



    X=X.unsqueeze(0)
    Y=Y.unsqueeze(0)
    
   
    p_x = ((X * filtered_im).sum(axis=(1,2)) / (mu_0+1e-10))
    p_y = ((Y * filtered_im).sum(axis=(1,2)) / (mu_0+1e-10))
    dx=p_x[1:]-dr[1:,0]-p_x[:-1]
    dy=p_y[1:]-dr[1:,1]-p_y[:-1]
    

    dx_tot=0
    dy_tot=0
    T=dx.shape[0]-(dt-1)
    for i in range(dt):
      dx_tot=dx_tot+dx[i:i+T]
      dy_tot=dy_tot+dy[i:i+T]


    speeds= torch.sqrt(  (dx_tot)**2  +  (dy_tot)**2  )/dt
     
    filter_d=((observations[t_1+dt:t_2,:,:,0].sum(axis=(1,2))>10 ) * (observations[t_1+dt:t_2,:,:,0].sum(axis=(1,2))<observations[0,:,:,0].sum(axis=(0,1))*3))
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

    if ((observations[-1,:,:,0].sum(axis=(0,1))>10 ) and (observations[-1,:,:,0].sum(axis=(0,1))<observations[0,:,:,0].sum(axis=(0,1))*3)):
      perf=perf+1/n
    else:
      fail.append(env)
    
    avg_speed=avg_speed+avg_speed_test(observations,system.config.SX,system.config.SY)/n
 
    

  
  return perf,fail,avg_speed



def try_robustness(system,crea,random_obstacles_pool,long_term=True):
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
  with torch.no_grad():
    observations=system.run()
  observations= observations.states
  speed=avg_speed_test(observations,system.config.SX,system.config.SY,t_1=25,t_2=50)
  print("speed=" +str(speed))
  collapse=(observations[:,:,:,0].sum(axis=(1,2))>10 )
  explode=(observations[:,:,:,0].sum(axis=(1,2))<observations[0,:,:,0].sum(axis=(0,1))*3)
  death=collapse*explode
  perf=death.detach().cpu().numpy().sum()

  performances.append(perf)

  
  speed_25_50_1=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=50,dt=1)
  speed_25_200_25=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=200,dt=25)
  speed_25_200_50=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=200,dt=50)
  speed_25_1000_50=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=1000,dt=50)
 
  ###   test surival random obstacles 
  perf=0
  n=len(random_obstacles_pool)
  perf,fails_env,avg_speed_rd=try_robustness_obstacles(system,crea,random_obstacles_pool)
  performances.append(perf)
  #fail.extend(fails_env)



  
  return performances,fail,avg_speed_rd,speed,speed_25_50_1,speed_25_200_25,speed_25_200_50,speed_25_1000_50
  
  



  

lenia_config = Old_lenia.default_config()
lenia_config.SX = 256
lenia_config.SY = 256
lenia_config.final_step = 50
lenia_config.version = 'pytorch_fft'

system = Old_lenia( config=lenia_config, device='cuda')






nb_env=50





p1=[]
p2=[]
p3=[]
p4=[]
p5=[]
list_speeds_rd=[]
list_avg_speed_rd=[]
list_speeds_hd=[]
list_avg_speed_hd=[]
speeds_25_1000_50=[]
speeds_25_200_25=[]
speeds_25_200_50=[]
speeds_25_50_1=[]


with open("../../../../creaManual/stats.json", 'r') as f:
  data = json.load(f)
nb=0
for key in data:
    if(data[key]["is_long_term_stable"] ):
        print(nb)
        nb=nb+1
        a=torch.load("../../../../creaManual/all/"+key+"_params.pickle")
        policy_parameters=a["policy_parameters"]
        
    
        system.reset(initialization_parameters=policy_parameters['initialization'],
        update_rule_parameters=policy_parameters['update_rule'])
        system.config.speed_x=0
            #if(speed>0.1):
        random_obstacles_pool=torch.zeros((nb_env,lenia_config.SX,lenia_config.SY))
        for i in range(nb_env):
          system.random_obstacle_bis(30)
          random_obstacles_pool[i]=system.init_wall*1.0
        random_obstacles_pool_bis=torch.zeros((nb_env,lenia_config.SX,lenia_config.SY))
        for i in range(nb_env):
          system.random_obstacle_bis(15)
          random_obstacles_pool_bis[i]=system.init_wall*1.0


        performances,_,avg_speed_rd,_,speed_25_50_1,speed_25_200_25,speed_25_200_50,speed_25_1000_50=try_robustness(system,policy_parameters,random_obstacles_pool)

        _,_,avg_speed_rd=try_robustness_obstacles(system,policy_parameters,random_obstacles_pool_bis)
        system.config.speed_x=1
        perf,_,_=try_robustness_obstacles(system,policy_parameters,random_obstacles_pool)
        system.config.speed_x=2
        perf2,_,_=try_robustness_obstacles(system,policy_parameters,random_obstacles_pool)
        p1.append(performances[0])
        p2.append(performances[1])
        #p3.append(performances[2])
        p4.append(perf)
        p5.append(perf2)
        speeds_25_50_1.append(speed_25_50_1)
        speeds_25_200_25.append(speed_25_200_25)
        speeds_25_200_50.append(speed_25_200_50)
        speeds_25_1000_50.append(speed_25_1000_50)
        print(key,performances,avg_speed_rd,perf,perf2,(speed_25_50_1,speed_25_200_25,speed_25_200_50,speed_25_1000_50))
        list_avg_speed_rd.append(avg_speed_rd)
    





np.save("exp"+str(expe)+"_rep"+str(seed)+"_Longstable.npy",p1)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_randomObs.npy",p2)
#np.save("exp"+str(expe)+"_rep"+str(seed)+"_hardObs.npy",p3)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_moveObs.npy",p4)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_move2Obs.npy",p5)
np.save("speeds_25_50_1.npy",speeds_25_50_1)
np.save("speeds_25_200_25.npy",speeds_25_200_25)
np.save("speeds_25_200_50.npy",speeds_25_200_50)
np.save("speeds_25_1000_50.npy",speeds_25_1000_50)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_speed_rd.npy",list_avg_speed_rd)


print("max speed (speeds_25_200_25) "+ str(np.max(speeds_25_200_25)))
print("max speed random "+ str(np.max(list_avg_speed_rd)))
#print("max speed hard "+ str(np.max(list_avg_speed_hd)))
print("max robust move2Obs"+str(np.max(p5)))
print("max robust moveObs"+str(np.max(p4)))
print("max robust rand"+str(np.max(p2)))
print("finished")

