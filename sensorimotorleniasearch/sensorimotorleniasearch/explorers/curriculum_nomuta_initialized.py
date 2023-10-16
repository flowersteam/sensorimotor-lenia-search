import torch
import random
import os
from addict import Dict
from sensorimotorleniasearch import Explorer
from sensorimotorleniasearch.spaces import BoxSpace
from sensorimotorleniasearch.utils import sample_value
import numbers
from tqdm import tqdm
from copy import deepcopy



class BoxGoalSpace(BoxSpace):
    def __init__(self, representation, autoexpand=True, low=0., high=0., shape=None, dtype=torch.float32):
        self.representation = representation
        self.autoexpand = autoexpand
        if shape is not None:
            if isinstance(shape, list) or isinstance(shape, tuple):
                assert len(shape) == 1 and shape[0] == self.representation.n_latents
            elif isinstance(shape, numbers.Number):
                assert shape == self.representation.n_latents
        BoxSpace.__init__(self, low=low, high=high, shape=(self.representation.n_latents,), dtype=dtype)

    def map(self, observations, **kwargs):
        embedding = self.representation.calc(observations, **kwargs)
        if self.autoexpand:
            embedding_c = embedding.detach()
            is_nan_mask = torch.isnan(embedding_c)
            if is_nan_mask.sum() > 0:
                embedding_c[is_nan_mask] = self.low[is_nan_mask]
                self.low = torch.min(self.low, embedding_c)
                embedding_c[is_nan_mask] = self.high[is_nan_mask]
                self.high = torch.max(self.high, embedding_c)
            else:
                self.low = torch.min(self.low, embedding_c)
                self.high = torch.max(self.high, embedding_c)
        return embedding

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        return self.representation.calc_distance(embedding_a, embedding_b, **kwargs)

    def sample(self):
        return BoxSpace.sample(self)


class CURIExplorer_nomuta_initialized(Explorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """

    # Set these in ALL subclasses
    goal_space = None  # defines the obs->goal representation and the goal sampling strategy (self.goal_space.sample())
    reach_goal_optimizer = None

    @staticmethod
    def default_config():
        default_config = Dict()
        # base config
        default_config.num_of_random_initialization = 40  # number of random runs at the beginning of exploration to populate the IMGEP memory

        # Pi: source policy parameters config
        default_config.source_policy_selection = Dict()
        default_config.source_policy_selection.type = 'optimal'  # either: 'optimal', 'random'

        # Opt: Optimizer to reach goal
        default_config.reach_goal_optimizer = Dict()
        default_config.reach_goal_optimizer.optim_steps = 10
        default_config.reach_goal_optimizer.name = "SGD"
        default_config.reach_goal_optimizer.initialization_cppn.parameters.lr =  1e-3
        default_config.reach_goal_optimizer.lenia_step.parameters.lr = 1e-4
        # default_config.reach_goal_optimizer.parameters.eps=1e-4

        return default_config

    def __init__(self, system, explorationdb, goal_space, config={}, **kwargs):
        super().__init__(system=system, explorationdb=explorationdb, config=config, **kwargs)

        self.goal_space = goal_space

        # initialize policy library
        self.policy_library = []

        # initialize goal library
        self.goal_library = torch.empty((0,) + self.goal_space.shape)

        # reach goal optimizer
        self.reach_goal_optimizer = None

    def get_source_policy_idx(self, target_goal):

        if self.config.source_policy_selection.type == 'optimal':
            # get distance to other goals
            tbis=self.goal_library*1.0
            #augment distance to creature that exploded or died because we don't want to select them.
            tbis[:,1]=tbis[:,1]+(tbis[:,1]<-9).float()*100
            tbis[:,1]=tbis[:,1]+(tbis[:,0]>0.11).float()*100
            goal_distances = self.goal_space.calc_distance(target_goal.unsqueeze(0), tbis)
            source_policy_idx = torch.argmin(goal_distances)

        elif self.config.source_policy_selection.type == 'random':
            source_policy_idx = sample_value(('discrete', 0, len(self.goal_library) - 1))

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection.type))

        return source_policy_idx
    
    def sample_further_goal(self):
      #change distance for reached goal when the creature died or exploded
      tbis=self.goal_library*1.0
      tbis[:,2]=tbis[:,2]+(tbis[:,1]<-9).float()*100
      tbis[:,2]=tbis[:,2]+(tbis[:,0]>0.11).float()*100

      target_goal=torch.ones(3)*-10
      #go a little further than previous best
      ind=torch.argmin(tbis[:,2])
      target_goal[0]=0.06
      target_goal[1]=tbis[ind,1]
      target_goal[2]=tbis[ind,2]-0.06
      return(target_goal)






    def run(self, n_exploration_runs,nb_obstacles=8, continue_existing_run=False):
      
      
        print('NEW TRY OF INIT')


        print('Exploration: ')
        progress_bar = tqdm(total=n_exploration_runs)
        if continue_existing_run:
            run_idx = len(self.policy_library)
            progress_bar.update(run_idx)
        else:
            self.policy_library = []
            self.goal_library = torch.empty((0,) + self.goal_space.shape)
            run_idx = 0
        
        ############# Beginning of the search ##############
        while run_idx < n_exploration_runs:
            policy_parameters = Dict.fromkeys(
                ['initialization', 'update_rule'])  # policy parameters (output of IMGEP policy)






            ############ Initial Random Sampling of Parameters ####################
            if len(self.policy_library) < self.config.num_of_random_initialization:

                target_goal = None
                source_policy_idx = None
                
                save=torch.load('run_{:07d}_data.pickle'.format(run_idx))
                policy_parameters=save['policy_parameters']
                # sample new parameters to test
                
                reached_goal = save["reached_goal"]
                observations=torch.zeros((50,self.system.config.SX,self.system.config.SY,2))
               
                optim_step_idx = 0
                dist_to_target = None
                





            ############## Goal-directed Sampling of Parameters ######################
            else:
                

                # sample a goal space from the goal space
                if(len(self.policy_library)-self.config.num_of_random_initialization<8):
                  target_goal=torch.ones(3)*-10
                  target_goal[0]=0.065
                  target_goal[2]=0.19-(len(self.policy_library)-self.config.num_of_random_initialization)*0.06
                  target_goal[1]=0
                else:
                  target_goal=self.sample_further_goal()
                                    
                if(len(self.policy_library)-self.config.num_of_random_initialization>=2):  
                  print(f'Run {run_idx}, optimisation toward goal: ')
                  print("TARGET ="+str(target_goal))

                
                # get source policy for this target goal
                source_policy_idx = self.get_source_policy_idx(target_goal)
                source_policy = self.policy_library[source_policy_idx]

                
                # if we're at the beginning or iteration%5==0 then don't mutate and train for longer
                if(len(self.policy_library)-self.config.num_of_random_initialization<8 or len(self.policy_library)%5==0 ):

                  policy_parameters['initialization'] = deepcopy(source_policy['initialization'])
                  policy_parameters['update_rule'] = deepcopy(source_policy['update_rule'])
                  self.system.reset(initialization_parameters=policy_parameters['initialization'],
                                    update_rule_parameters=policy_parameters['update_rule'])
                  ite=self.config.reach_goal_optimizer.optim_steps
                # else mutate 
                else:
                  ite=15
                  # mutate until finding a non dying and non exploding creature
                  
                  policy_parameters['initialization'] = deepcopy(source_policy['initialization'])
                  policy_parameters['update_rule'] = deepcopy(source_policy['update_rule'])
                  self.system.reset(initialization_parameters=policy_parameters['initialization'],
                                    update_rule_parameters=policy_parameters['update_rule'])
                  

                ##### INNER LOOP (Optimization part toward target goal ) ####
                if isinstance(self.system, torch.nn.Module) and self.config.reach_goal_optimizer.optim_steps > 0:


                    optimizer_class = eval(f'torch.optim.{self.config.reach_goal_optimizer.name}')
                    self.reach_goal_optimizer = optimizer_class([{'params': self.system.initialization.parameters(), **self.config.reach_goal_optimizer.initialization_cppn.parameters},
                                                                {'params': self.system.lenia_step.parameters(), **self.config.reach_goal_optimizer.lenia_step.parameters}],
                                                                **self.config.reach_goal_optimizer.parameters)
                    
                    last_dead=False
                    for optim_step_idx in range(1, ite):
                        
                        # run system with IMGEP's policy parameters
                        self.system.random_obstacle(nb_obstacles)
                        self.system.generate_init_state()
                        observations = self.system.run()
                        reached_goal = self.goal_space.map(observations)
                        
                        ### Define  target disk
                        x = torch.arange(self.system.config.SX)
                        y = torch.arange(self.system.config.SY)
                        xx = x.view(-1, 1).repeat(1, self.system.config.SY)
                        yy = y.repeat(self.system.config.SX, 1)
                        X = (xx -(target_goal[1]+0.5)*self.system.config.SX).float() / (35)
                        Y = (yy - (target_goal[2]+0.5)*self.system.config.SY).float() / (35)
                        # distance to center in normalized space
                        D = torch.sqrt(X ** 2 + Y ** 2)
                        # mask is the target circles
                        mask=0.85*(D<0.5).float()+0.15*(D<1).float()



                        loss= (0.9*mask-observations.states[-1,:,:,0]).pow(2).sum().sqrt() 

                        # optimisation step
                        self.reach_goal_optimizer.zero_grad()
                        loss.backward()
                        self.reach_goal_optimizer.step()
                       


                        #compute again the kernels for the next step because parameters have been changed with the optimization
                        self.system.lenia_step.compute_kernel()


                        dead=observations.states[-1,:,:,0].sum()<10
                        if(dead and last_dead):
                          self.reach_goal_optimizer.zero_grad()
                          break
                        last_dead=dead

                    ###### END of INNER loop #####



                    # gather back the trained parameters
                    self.system.update_initialization_parameters()
                    self.system.update_update_rule_parameters()
                    policy_parameters['initialization'] = self.system.initialization_parameters
                    policy_parameters['update_rule'] = self.system.update_rule_parameters
                    dist_to_target = loss.item()

                    
                      

                ## look at the reached goal ##
                reached_goal=torch.zeros(3).cpu()
                with torch.no_grad():
                  for i in range(20):
                    self.system.random_obstacle(nb_obstacles)
                    self.system.generate_init_state()
                    observations = self.system.run()
                    if(observations.states[-1,:,:,0].sum()<10):
                      reached_goal[0]=10
                      break
                    reached_goal = reached_goal+self.goal_space.map(observations).cpu()/20
                if(len(self.policy_library)-self.config.num_of_random_initialization>=2):
                  print("reached= "+str(reached_goal))


                


            # save results
            reached_goal=reached_goal.cpu()
            self.db.add_run_data(id=run_idx,
                                 policy_parameters=policy_parameters,
                                 observations=observations,
                                 source_policy_idx=source_policy_idx,
                                 target_goal=target_goal,
                                 reached_goal=reached_goal,
                                 n_optim_steps_to_reach_goal=optim_step_idx,
                                 dist_to_target=dist_to_target)

            # add policy and reached goal into the libraries
            # do it after the run data is saved to not save them if there is an error during the saving
           
            
            self.policy_library.append(policy_parameters)
            self.goal_library = torch.cat([self.goal_library, reached_goal.reshape(1, -1).to(self.goal_library.device).detach()])
            #if len(self.policy_library) >= self.config.num_of_random_initialization:
              #plt.imshow(self.system.init_wall.cpu())
              #plt.scatter(((self.goal_library[:,0]<0.11).float()*(self.goal_library[:,2]>-0.5).float()*(self.goal_library[:,2]+0.5)*self.system.config.SY).cpu(),((self.goal_library[:,0]<0.11).float()*(self.goal_library[:,1]>-0.5).float()*(self.goal_library[:,1]+0.5)*self.system.config.SX).cpu())
              #plt.show()
            # increment run_idx


            run_idx += 1
            progress_bar.update(1)


            #Cases where we need to try again with new init (those are arbitrary to go faster)

            
            


            if len(self.policy_library)==n_exploration_runs-1:
              again=False


                
            
          
                
                
              
