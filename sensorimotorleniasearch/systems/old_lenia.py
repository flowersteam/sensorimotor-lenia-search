
from scipy import ndimage
from addict import Dict
import torch
from sensorimotorleniasearch.utils import roll_n
from sensorimotorleniasearch.spaces import Space,DiscreteSpace,BoxSpace,DictSpace,MultiDiscreteSpace
import numpy as np
from copy import deepcopy
import warnings


## Systems

bell = lambda x, m, s: torch.exp(-((x-m)/s)**2 / 2) 
# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda u: (4 * u * (1 - u)) ** 4,  # polynomial (quad4)
    1: lambda u: torch.exp(4 - 1 / (u * (1 - u))),  # exponential / gaussian bump (bump4)
    2: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float(),  # step (stpz1/4)
    3: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float() + (u < q).float() * 0.5,  # staircase (life)
    4: lambda u: torch.exp(-(u-0.5)**2/0.2),
    8: lambda u: (torch.sin(10*u)+1)/2,
    9: lambda u: (a*torch.sin((u.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2,
}





field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
    3: lambda n, m, s: - torch.clamp(n-m,0,1)*s #food eating kernl
}

# ker_c =lambda r,a,b,c :(a*torch.sin((r.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2
ker_c= lambda x,r,w,b : (b*torch.exp(-((x.unsqueeze(-1)-r)/w)**2 / 2)).sum(-1) 

class Dummy_init_mod(torch.nn.Module):
  def __init__(self,init):
    torch.nn.Module.__init__(self)
    self.register_parameter('init', torch.nn.Parameter(init))

# Lenia Step FFT version (faster)
class LeniaStepFFT_old(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(self,C, R, T, b,r,h, m, s, kn, gn, nb_k=1, is_soft_clip=False, SX=256, SY=256,speed_x=0,speed_y=0,device='cpu'):
        super(LeniaStepFFT_old, self).__init__()
        # Lenia Parameters: 
        ## R is not differentiable
        self.C=C
        self.R = torch.tensor(R).to(device)+15
        self.h=torch.tensor(h)
        ## in original Lenia T is an integer but we consider float to add it to the list of differentiable parameters
        self.T =torch.tensor(T).to(device)
        ## in original Lenia b is a vector which length can be 1,2 or 3 (up to 3 concentric rings) 
        ## but we fix the length to 4 and consider than if b[i] < threshold then there is no ring to add it to the list of differentiable parameters
        self.b = torch.tensor(b).to(device)
        self.r = torch.tensor(r)
        self.m = torch.tensor(m)
        self.s = torch.tensor(s)
        self.kn = kn
        self.gn = gn
        self.nb_k=nb_k

        self.SX = SX
        self.SY = SY
        self.is_soft_clip = is_soft_clip 
        self.device=device
        self.speed_x=speed_x
        self.speed_y=speed_y
        self.compute_kernel()
        self.compute_kernel_env()

    def compute_kernel(self):
        self.kernels=torch.zeros((self.nb_k,self.SX,self.SY//2+1))
        # implementation of meshgrid in torch
        x = torch.arange(self.SX)
        y = torch.arange(self.SY)
        xx = x.repeat(self.SY, 1)
        yy = y.view(-1, 1).repeat(1, self.SX)
        
        
        for k in range(self.nb_k):
            X = (xx - int(self.SX / 2)).float() / (self.R*self.r[k])
            Y = (yy - int(self.SY / 2)).float() / (self.R*self.r[k])

            # distance to center in normalized space
            D = torch.sqrt(X ** 2 + Y ** 2)
            kfunc = kernel_core[self.kn[k].item()]
            # kernel
            nbumps = (self.b[k] >= 0).sum()  # modification to allow b always of length 4
            kr = nbumps * D
            b = self.b[k][torch.min(torch.floor(kr).long(), (nbumps - 1) * torch.ones_like(kr).long())]
            kernel = (D < 1).double() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
            kernel_sum = torch.sum(kernel)
            # normalization of the kernel
            self.kernel_norm = (kernel / kernel_sum)

            #print(self.kernel_norm.shape)
            # fft of the kernel

            kernel_FFT = torch.fft.rfftn(self.kernel_norm, dim=(0,1)).to(self.device)
            
            self.kernels[k]=kernel_FFT
            

    def compute_kernel_env(self):
      """ computes the kernel and the kernel FFT of the environnement from the parameters"""
      x = torch.arange(self.SX).to(self.device)
      y = torch.arange(self.SY).to(self.device)
      xx = x.view(-1, 1).repeat(1, self.SY)
      yy = y.repeat(self.SX, 1)
      X = (xx - int(self.SX / 2)).float() 
      Y = (yy - int(self.SY / 2)).float() 
      D = torch.sqrt(X ** 2 + Y ** 2)/(4)
      kernel = torch.sigmoid(-(D-1)*10) * ker_c(D,torch.tensor(np.array([0,0,0])).to(self.device),torch.tensor(np.array([0.5,0.1,0.1])).to(self.device),torch.tensor(np.array([1,0,0])).to(self.device))
      kernel_sum = torch.sum(kernel)
      kernel_norm = (kernel / kernel_sum).unsqueeze(0)
    
      kernel_FFT = torch.fft.rfftn(kernel_norm, dim=(-2,-1)).to(self.device)
      self.kernel_wall=kernel_FFT


    def forward(self, input):
        
        
        input[:,:,:,1]=torch.roll(input[:,:,:,1],[self.speed_y,self.speed_x],[1,2])
        
        self.D=torch.zeros(input.shape).to(self.device)
        self.Dn=torch.zeros(self.C)

        world_FFT = [torch.fft.rfftn(input[:,:,:,i],dim=(-2,-1)) for i in range(self.C)]
        

        ## speed up of the update for 1 channel creature by multiplying by all the kernel FFT  


        
        world_FFT_c = world_FFT[0]
        #multiply the FFT of the world and the kernels

        potential_FFT = self.kernels* world_FFT_c
        #ifft + realign 
        potential = torch.fft.irfftn(potential_FFT, dim=(-2,-1))
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        #growth function
        field=torch.zeros((self.nb_k,self.SX,self.SY))
        for k in range(self.nb_k):
            gfunc = field_func[min(self.gn[k].item(), 3)]
            
            field[k] = gfunc(potential[k], self.m[k], self.s[k])
        #add the growth multiplied by the weight of the rule to the total growth 
        
        self.D[:,:,:,0]=(self.h[:self.nb_k].unsqueeze(-1).unsqueeze(-1)*field).sum(0,keepdim=True)
        self.Dn[0]=self.h[:self.nb_k].sum()
        
        #fields = [field_func[self.gn[k].item()](potential[k], self.m[k], self.s[k]) for k in range(self.nb_k)]

        #for k in range(self.nb_k):
        #    self.D[:,:, :,0] = self.D[:,:, :,0] + self.h[k] * fields[k]

        #apply wall
        world_FFT_c = world_FFT[self.C-1]

        self.potential_FFT =  self.kernel_wall*world_FFT_c
        self.potential = torch.fft.irfftn(self.potential_FFT, dim=(-2,-1))
        self.potential = roll_n(self.potential, 2, self.potential.size(2) // 2)
        self.potential = roll_n(self.potential, 1, self.potential.size(1) // 2)


        gfunc = field_func[3]
        self.field = gfunc(self.potential, 1e-8, 10)
        for i in range(self.C-1):
          c1b=i
          self.D[:,:,:,c1b]=self.D[:,:,:,c1b]+1*self.field
          self.Dn[c1b]=self.Dn[c1b]+1

        
        ## Add the total growth to the current state 
        if not self.is_soft_clip:
            
            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0., max=1.)
            # output_img = input + (1.0 / self.T) * ((self.D/self.Dn+1)/2-input)
           
        else:
            output_img = torch.sigmoid((input + (1.0 / self.T) * self.D-0.5)*10)
             # output_img = torch.tanh(input + (1.0 / self.T) * self.D)
            

        return output_img
    
    
    

class Old_lenia(torch.nn.Module):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.version = 'pytorch_fft'  # "pytorch_fft", "pytorch_conv2d"
        default_config.SX = 256
        default_config.SY = 256
        default_config.final_step = 50
        default_config.C = 2
        default_config.speed_x=0
        default_config.speed_y=0
        return default_config


    def __init__(self, nb_k=10,init_size=40, config={}, device=torch.device('cpu'), **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        torch.nn.Module.__init__(self)
        self.device = device
        self.init_size=init_size
        if(device=="cuda"):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.run_idx = 0
        self.init_wall=torch.zeros((self.config.SX,self.config.SY))
        #reset with no argument to sample random parameters
        self.to(self.device)


    def reset(self, initialization_parameters, update_rule_parameters):
        # call the property setters
        self.initialization_parameters = initialization_parameters

        self.update_rule_parameters = update_rule_parameters
        
        # initialize Lenia CA with update rule parameters
        if self.config.version == "pytorch_fft":
            lenia_step = LeniaStepFFT_old(self.config.C,self.update_rule_parameters['R'], self.update_rule_parameters['T'],self.update_rule_parameters['b'], self.update_rule_parameters['r'],self.update_rule_parameters['h'], self.update_rule_parameters['m'],self.update_rule_parameters['s'],self.update_rule_parameters['kn'],self.update_rule_parameters['gn'],nb_k=self.update_rule_parameters['m'].shape[0], is_soft_clip=False, SX=self.config.SX, SY=self.config.SY,speed_x=self.config.speed_x,speed_y=self.config.speed_y,device=self.device)
        self.add_module('lenia_step', lenia_step)

        # initialize Lenia initial state with initialization_parameters
        init = self.initialization_parameters['init']
        # initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.initialization_space.config.neat_config, device=self.device)
        self.add_module('initialization', Dummy_init_mod(init))
        
        # push the nn.Module and the available device
        self.to(self.device)
        self.generate_init_state()
    
    def  random_obstacle(self,nb_obstacle=6,radius_obstacle=10):
      self.init_wall=torch.zeros((self.config.SX,self.config.SY))

      x = torch.arange(self.config.SX)
      y = torch.arange(self.config.SY)
      xx = x.view(-1, 1).repeat(1, self.config.SY)
      yy = y.repeat(self.config.SX, 1)
      for i in range(nb_obstacle):        
        X = (xx - int(torch.rand(1)*self.config.SX )).float() 
        Y = (yy - int(torch.rand(1)*self.config.SY/2)).float() 
        D = torch.sqrt(X ** 2 + Y ** 2)/radius_obstacle
        mask=(D<1).float()
        self.init_wall=torch.clamp(self.init_wall+mask,0,1)

    def  random_obstacle_bis(self,nb_obstacle=6,radius_obstacle=10,pos_obs=None):
      self.init_wall=torch.zeros((self.config.SX,self.config.SY))
      
      x = torch.arange(self.config.SX)
      y = torch.arange(self.config.SY)
      xx = x.view(-1, 1).repeat(1, self.config.SY)
      yy = y.repeat(self.config.SX, 1)
      for i in range(nb_obstacle):        
        X = (xx - int(torch.rand(1)*self.config.SX )).float() 
        Y = (yy - int(torch.rand(1)*self.config.SY)).float() 
        D = torch.sqrt(X ** 2 + Y ** 2)/radius_obstacle
        mask=(D<1).float()
        self.init_wall=torch.clamp(self.init_wall+mask,0,1)
        
      if( pos_obs is not None):
        X = (xx - pos_obs[0]).float() 
        Y = (yy - pos_obs[1]).float() 
        D = torch.sqrt(X ** 2 + Y ** 2)/radius_obstacle
        mask=(D<1).float()
        self.init_wall=torch.clamp(self.init_wall+mask,0,1)
        
      blur_radius = 10
      threshold = 0.05
      # smooth the image (to remove small objects)
      init_state = torch.zeros( self.config.SX, self.config.SY, dtype=torch.float64)
      sx, sy = self.initialization.init.shape
      init_hoffset=10
        
      init_state[self.config.SX // 2 - sx // 2:self.config.SX // 2 - sx // 2 + sx,init_hoffset:init_hoffset+sy] = torch.flip(self.initialization.init,dims=(1,))

      imgf = ndimage.gaussian_filter(init_state.detach().cpu().numpy(), blur_radius)
        

      # protect from wall
      self.init_wall[ imgf > 0] = 0.0


        

    def generate_init_state(self,X=105,Y=180):
        init_state = torch.zeros( 1,self.config.SX, self.config.SY,self.config.C, dtype=torch.float64)
        sx, sy = self.initialization.init.shape
        init_hoffset=10
        init_state[:, self.config.SX // 2 - sx // 2:self.config.SX // 2 - sx // 2 + sx,
            +init_hoffset:+init_hoffset+sy,0] = torch.flip(self.initialization.init,dims=(1,))
        if(self.config.C>1):
          init_state[0,:,:,1]=self.init_wall
        self.state = init_state.to(self.device)
        self.step_idx = 0



    def step(self, intervention_parameters=None):
        self.state = self.lenia_step(self.state)
        self.step_idx += 1
        return self.state


    def forward(self):
        state = self.step(None)
        return state


    def run(self): 
        """ run lenia for the number of step specified in the config.
        Returns the observations containing the state at each timestep"""
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.states = torch.empty((self.config.final_step, self.config.SX, self.config.SY,self.config.C))
        observations.states[0]  = self.state
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.states[step_idx] = cur_observation[0,:,:,:]
        return observations

    def save(self, filepath):
        """
        Saves the system object using torch.save function in pickle format
        Can be used if the system state's change over exploration and we want to dump it
        """
        torch.save(self, filepath)
    

    def close(self):
        pass


