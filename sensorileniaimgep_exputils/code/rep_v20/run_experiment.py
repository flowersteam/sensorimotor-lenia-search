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

sf="../../experiment_00000"+str(expe)+"/repetition_00000"+str(seed)  


def get_contour( image,threshold=0.1):

        kernel = np.ones((7,7),np.float32)/49
        image = cv2.filter2D(image,-1,kernel)

        
        #in case creature is on the cyclic border
        if(image[:,-1].sum()>0 and image[:,0].sum()>0):
            image[:,:30]=0
        if(image[-1,:].sum()>0 and image[0,:].sum()>0):
            image[:30,:]=0

        #plt.imshow(image)
        #plt.show()
        image_8bit = np.uint8(image * 255)
        

        threshold_level = threshold
        _, binarized = cv2.threshold(image_8bit, threshold_level, 255, cv2.THRESH_BINARY)

        # Find the contours of a binary image using OpenCV.

        contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       
        #print(len(contours))
        #print(hierarchy)
        

        return len(contours)
    

def avg_speed_test(observations,SX,SY,t_1=250,t_2=500,dt=10):
    
    x = torch.arange(SX)
    y = torch.arange(SY)
    yy = y.repeat(SX, 1)
    xx = x.view(-1, 1).repeat(1, SY)
    X = (xx ).double()
    Y = (yy ).double()
    
    
    filtered_im =observations[t_1:t_2,:,:,0]
    
    X=X.unsqueeze(0)
    Y=Y.unsqueeze(0)
    mu_0 = filtered_im.sum(axis=(1,2))
    p_x = ((X * filtered_im).sum(axis=(1,2)) / (mu_0+1e-10))
    p_y = ((Y * filtered_im).sum(axis=(1,2)) / (mu_0+1e-10))
    dx=p_x[dt:]-p_x[:-dt]
    dy=p_y[dt:]-p_y[:-dt]
    dx=dx- (dx>40)*256 +(dx<-40)*256 
    dy=dx- (dy>40)*256 +(dy<-40)*256
    speeds= torch.sqrt(  (dx)**2  +  (dy)**2  )
     
    filter_d=((observations[t_1+dt:t_2,:,:,0].sum(axis=(1,2))>10 ) * (observations[t_1+dt:t_2,:,:,0].sum(axis=(1,2))<observations[0,:,:,0].sum(axis=(0,1))*3))
    speeds=speeds *filter_d
 
    avg_speed = speeds.sum()/(dt*speeds.shape[0])
    
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



def try_robustness(system,crea,random_obstacles_pool,hard_env_pool,long_term=True):
  system.reset(initialization_parameters=crea['initialization'],
                update_rule_parameters=crea['update_rule'])
  performances=[]
  fail=[]
  speed=0
  #### test survival without obstacles
  if(long_term):
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
      nb_contours=get_contour( observations[-1,:,:,0].cpu().numpy(),threshold=0.1)
      if(nb_contours>1):
        perf=0
      performances.append(perf)
    
  else: 
    performances.append(0)
    perf=0
  
  speed_25_50_1=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=50,dt=1)
  speed_25_200_25=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=200,dt=25)
  speed_25_200_50=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=200,dt=50)
  speed_25_1000_50=avg_speed_test(observations,lenia_config.SX,lenia_config.SY,t_1=25,t_2=1000,dt=50)
 
  if(perf>999):  
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
  else:
      performances.append(0)
      performances.append(0)
      avg_speed_rd=0
      avg_speed_hd=0
  #fail.extend(fails_env)
  
  
  return performances,fail,avg_speed_rd,avg_speed_hd,speed,speed_25_50_1,speed_25_200_25,speed_25_200_50,speed_25_1000_50
  
  



  

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
lenia_config.speed_x=1
lenia_config.speed_y=0

initialization_space_config = Dict()
initialization_space = LeniaInitializationSpace(config=initialization_space_config)

system_move = Lenia_C_move(initialization_space=initialization_space, config=lenia_config, device='cuda')




nb_env=50
random_obstacles_pool=torch.zeros((nb_env,lenia_config.SX,lenia_config.SY))


for i in range(nb_env):
  system.random_obstacle_bis(30)
  random_obstacles_pool[i]=system.init_wall*1.0

hard_env_pool=torch.load("../../../hand_obstacles.pickle")



random_obstacles_pool_bis=torch.zeros((nb_env,lenia_config.SX,lenia_config.SY))
for i in range(nb_env):
  system.random_obstacle_bis(15)
  random_obstacles_pool_bis[i]=system.init_wall*1.0




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
creas=[]
count=0
for i in range(30000):
    
 
    policy_parameters = Dict.fromkeys(['initialization', 'update_rule']) 
    policy_parameters['initialization'] = system.initialization_space.sample()
    policy_parameters['update_rule'] = system.update_rule_space.sample()
    #divide h by 3 at the beginning as some unbalanced kernels can easily kill
    policy_parameters['update_rule'].h =policy_parameters['update_rule'].h*3
    system.reset(initialization_parameters=policy_parameters['initialization'],
    update_rule_parameters=policy_parameters['update_rule'])
    
    
    
    #run the system
    with torch.no_grad():
        system.config.final_step = 50
        system.random_obstacle(0)
        system.generate_init_state()
        observations = system.run()
        observations=observations.states
        slp=(observations[-1,:,:,0].sum(axis=(0,1))>10 ) and (observations[-1,:,:,0].sum(axis=(0,1))<observations[0,:,:,0].sum(axis=(0,1))*3)
        #reached_goal = goal_space.map(observations)
     #is_dead= reached_goal[0]>0.09 or reached_goal[1]<-0.5
    nb_contours=get_contour( observations[-1,:,:,0].cpu().numpy(),threshold=0.1)
  
    if(slp and (nb_contours<2)):
        
        

        #if(speed>0.1): 
        performances,_,avg_speed_rd,avg_speed_hd,_,speed_25_50_1,speed_25_200_25,speed_25_200_50,speed_25_1000_50=try_robustness(system,policy_parameters,random_obstacles_pool,hard_env_pool)
        
        #else:
            #performances=[-1,1.,1.]
            #avg_speed_hd=speed
            #avg_speed_rd=speed
        if(performances[0]>999):
            print(nb_contours)
            torch.save(policy_parameters,str(count)+".pickle")
            count=count+1
            #creas.append(policy_parameters)
            _,_,avg_speed_rd=try_robustness_obstacles(system,policy_parameters,random_obstacles_pool_bis)
            system_move.config.speed_x=1
            perf,_,_=try_robustness_obstacles(system_move,policy_parameters,random_obstacles_pool)
            system_move.config.speed_x=2
            perf2,_,_=try_robustness_obstacles(system_move,policy_parameters,random_obstacles_pool)
            p1.append(performances[0])
            p2.append(performances[1])
            p3.append(performances[2])
            p4.append(perf)
            p5.append(perf2)
            speeds_25_50_1.append(speed_25_50_1)
            speeds_25_200_25.append(speed_25_200_25)
            speeds_25_200_50.append(speed_25_200_50)
            speeds_25_1000_50.append(speed_25_1000_50)
            print(i,performances,avg_speed_rd,avg_speed_hd,perf,perf2,(speed_25_50_1,speed_25_200_25,speed_25_200_50,speed_25_1000_50))
            list_avg_speed_rd.append(avg_speed_rd)
            list_avg_speed_hd.append(avg_speed_hd)
            #np.save("ob"+str(i)+".npy",observations.cpu().numpy())
    


list_avg_speed_rd=np.array(list_avg_speed_rd)
list_avg_speed_hd=np.array(list_avg_speed_hd)




np.save("exp"+str(expe)+"_rep"+str(seed)+"_Longstable.npy",p1)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_randomObs.npy",p2)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_hardObs.npy",p3)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_moveObs.npy",p4)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_move2Obs.npy",p5)
np.save("speeds_25_50_1.npy",speeds_25_50_1)
np.save("speeds_25_200_25.npy",speeds_25_200_25)
np.save("speeds_25_200_50.npy",speeds_25_200_50)
np.save("speeds_25_1000_50.npy",speeds_25_1000_50)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_speed_rd.npy",list_avg_speed_rd)
np.save("exp"+str(expe)+"_rep"+str(seed)+"_speed_hd.npy",list_avg_speed_hd)


#torch.save(creas,"creas.pickle")
print("max speed random "+ str(np.max(list_avg_speed_rd)))
print("max speed hard "+ str(np.max(list_avg_speed_hd)))
print("max robust moveObs"+str(np.max(p4)))
print("max robust rand"+str(np.max(p2)))
print("finished")

