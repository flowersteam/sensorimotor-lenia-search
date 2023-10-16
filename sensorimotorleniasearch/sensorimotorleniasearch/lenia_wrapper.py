from addict import Dict


from sensorimotorleniasearch.systems import LeniaStepFFTC, Lenia_C, kernel_core, field_func

from sensorimotorleniasearch.spaces import Space,DiscreteSpace,BoxSpace,DictSpace,MultiDiscreteSpace



import imageio
import io
from matplotlib import colors
import math
import numpy as np
import os
from PIL import Image, ImageDraw
import scipy
import torch
import torchvision.transforms.functional as Fvision

class TmpLeniaStepFFT(LeniaStepFFTC):

    def compute_kernel(self):
        self.size=(self.SX,self.SY)
        if torch.__version__ >= "1.7.1":
            self.kernels = torch.zeros((self.nb_k, *[s for s in self.size[:-1]], self.size[-1] // 2 + 1),
                                       device=self.R.device)
        else:
            self.kernels = torch.zeros((self.nb_k, *self.size, 2), device=self.R.device)

        dims = [slice(0, s) for s in self.size]
        I = list(reversed(np.mgrid[list(reversed(dims))]))  # I, J, K, L
        MID = [int(s / 2) for s in self.size]
        X = [torch.from_numpy(i - mid).to(self.R.device) for i, mid in
             zip(I, MID)]  # X, Y, Z, S #TODO: removed division by self.R

        for k in range(self.nb_k):
            # distance to center in normalized space
            D = torch.sqrt(torch.stack([x ** 2 for x in X]).sum(axis=0)) / ((self.R+15) * self.r[k])
            # kernel
            kfunc = kernel_core[self.kn[k].item()]
            if self.kn[k].item() == 4:
                kernel = torch.sigmoid(-(D - 1) * 10) * kfunc(D, self.rk[k], self.w[k], self.b[k])  # -(D-1)*10 to have a sigmoid that goes from 1 (D=0) to 0 (vertical asymptop to D=1 more or less smooth based on float value here 10)
            else:
                nbumps = (self.b[k] >= 0).sum()  # modification to allow b always of length 4
                kr = nbumps * D
                b = self.b[k][torch.min(torch.floor(kr).long(), (nbumps - 1) * torch.ones_like(kr).long())]
                kernel = (D < 1).double() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
            kernel_sum = torch.sum(kernel)
            kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
            # fft of the kernel
            if torch.__version__ >= "1.7.1":
                kernel_FFT = torch.fft.rfftn(kernel_norm, dim=tuple(range(-len(self.size), 0)))
            else:
                kernel_FFT = torch.rfft(kernel_norm, signal_ndim=len(self.size), onesided=False)
            self.kernels[k] = kernel_FFT
        self.kernels.to(self.device)


            

class TmpLenia(Lenia_C):
    
   

    def reset(self, run_parameters=None):
        if(run_parameters is not None):
            
            init_state = torch.zeros(1, *self.config.size,self.config.C-1, dtype=torch.float64, device=run_parameters.init_state.device)
            init_mask =  [0]+ [slice(s//2 - math.ceil(run_parameters.init_state.shape[i]/2), s//2+run_parameters.init_state.shape[i]//2) for i,s in enumerate(self.config.size)] +[ 0]
            init_state[init_mask] = run_parameters.init_state
            self.state = init_state
            # wall init
            if self.config.wall_c:
                init_wall = run_parameters.init_wall.unsqueeze(0).unsqueeze(-1)
                init_wall[init_wall<0.7] = 0.0
                self.state = torch.cat([self.state, init_wall], 3)
            else:
                init_wall = torch.zeros(1, *self.config.size,1,dtype=torch.float64, device=run_parameters.init_state.device)
                self.state = torch.cat([self.state, init_wall], 3)
                 
            	
                
            self.lenia_step = TmpLeniaStepFFT(C=self.config.C, SX=self.config.size[0],SY=self.config.size[1],
                                           R=run_parameters.R, T=run_parameters.T,
                                           c0=run_parameters.c0, c1=run_parameters.c1,
                                           r=run_parameters.r, rk=run_parameters.rk,
                                           b=run_parameters.b, w=run_parameters.w, h=run_parameters.h,
                                           m=run_parameters.m, s=run_parameters.s,
                                           kn=run_parameters.kn, gn=run_parameters.gn,device=self.device)

            self._observations = Dict()
            self._observations.states = torch.empty((self.config.final_step, *self.config.size, self.config.C), device=self.state.device)
            self._observations.states[0] = self.state[0]

            self.step_idx = 0

            current_observation = Dict()
            current_observation.state = self._observations.states[0]

            return current_observation
        else :
            return

    def reset_from_crea_filepath(self, crea_filepath, scaling=1.0, init_hoffset=10, init_rot_angle=0., init_hflip=True,
                                init_noise_rate=0.0, init_noise_std=0.0, obstacle_number=0, obstacle_radius=10, obstacle_protection="blurr+10",
                                obstacle_positions=None, device="cuda"):
        self.device=device
        self.config.size=(self.config.SX,self.config.SY)
        crea = torch.load(crea_filepath)
        crea_parameters = Dict()
        
        self.initialization=crea['policy_parameters']['initialization']
        self.update_rule=crea['policy_parameters']['update_rule']
        # INIT STATE
        init_patch = crea['policy_parameters']['initialization']['init'].unsqueeze(0).to(device)
        #self.initialization=crea['policy_parameters']['initialization']
        ## resize
        if scaling != 1.0:
            init_size = tuple([int(s * scaling) for s in init_patch.shape[1:]])
            init_patch = Fvision.resize(init_patch, init_size, interpolation=Fvision.InterpolationMode.BILINEAR, antialias=True)
        else:
            init_size = init_patch.shape[1:]

        ## add noise
        if init_noise_rate > 0.0:
            noise_mask = (torch.rand(init_size) < init_noise_rate).type(init_patch.dtype).to(init_patch.device)
            if init_noise_std > 0.0:
                noise_values = init_noise_std * torch.randn(init_size).type(init_patch.dtype).to(init_patch.device)
                init_patch = init_patch + noise_mask * noise_values

        ## put init in grid
        crea_parameters.init_state = torch.zeros((self.config.SX,self.config.SY), dtype=crea['policy_parameters']['initialization']['init'].dtype).to(device)
        sy, sx = init_size
        SY, SX = self.config.size
        
        if init_hflip: #crea going to the left so we initialize the square on the right of the grid
            crea_parameters.init_state[ SY // 2 - sy // 2:SY // 2 - sy // 2 + sy,
            -init_hoffset - sx:-init_hoffset] = init_patch[0]
        else:
            crea_parameters.init_state[ SY // 2 - sy // 2:SY // 2 - sy // 2 + sy,
            init_hoffset:init_hoffset+sx] = init_patch[0]

        ## rotate
        if init_rot_angle > 0.0:
            com = (SY // 2, -init_hoffset - sx // 2)
            crea_parameters.init_state = Fvision.rotate(crea_parameters.init_state, init_rot_angle, center=com,
                                                        interpolation=Fvision.InterpolationMode.BILINEAR)

        ## horizontal flip
        if init_hflip:
            crea_parameters.init_state = Fvision.hflip(crea_parameters.init_state)

        # WALLS
        if self.config.wall_c:
            crea_parameters.init_wall = torch.zeros((self.config.SX,self.config.SY), dtype=crea['policy_parameters']['initialization']['init'].dtype).to(device)
            obstacle_number = int(obstacle_number)

            if obstacle_number > 0:
                x = torch.arange(self.config.size[0])
                y = torch.arange(self.config.size[1])
                xx = x.view(-1, 1).repeat(1, self.config.size[1])
                yy = y.repeat(self.config.size[0], 1)

            if obstacle_positions is not None:
                for obs_pos in obstacle_positions:
                    X = (xx - obs_pos[0]).float()
                    Y = (yy - obs_pos[1]).float()
                    D = torch.sqrt(X ** 2 + Y ** 2) / obstacle_radius
                    mask = (D < 1).float().to(device)
                    crea_parameters.init_wall = torch.clamp(crea_parameters.init_wall + mask, 0, 1)

                obstacle_number -= obstacle_positions.shape[0]

            for i in range(obstacle_number):
                X = (xx - int(torch.rand(1) * self.config.size[0])).float()
                Y = (yy - int(torch.rand(1) * self.config.size[1])).float()
                D = torch.sqrt(X ** 2 + Y ** 2) / obstacle_radius
                mask = (D < 1).float().to(device)
                crea_parameters.init_wall = torch.clamp(crea_parameters.init_wall + mask, 0, 1)

            if obstacle_protection == "square+10":
                crea_parameters.init_wall[ SY//2-sy//2-10:SY//2-sy//2+sy+10, init_hoffset-10:init_hoffset+sx+10] = 0.0
            elif obstacle_protection == "blurr+10":
                blur_radius = 10
                threshold = 0.05
                # smooth the image (to remove small objects)
                imgf = scipy.ndimage.gaussian_filter(crea_parameters.init_state.cpu(), blur_radius)
                # protect from wall
                crea_parameters.init_wall[ imgf > threshold] = 0.0

        # UPDATE SYSTEM WITH APPROPRIATE NUMBER OF KERNELS
        nb_k = len(crea['policy_parameters']['update_rule']['c0'])
        self.config.nb_k = 1
        self.update_rule_space.spaces["c0"] = MultiDiscreteSpace(nvec=[self.config.C] * self.config.nb_k,mutation_mean=0.0, mutation_std=0.1, indpb=0.1)
        #self.update_rule_space.spaces["c0"].initialize(self)
        self.update_rule_space.spaces["c1"] = MultiDiscreteSpace(nvec=[self.config.C] * self.config.nb_k,mutation_mean=0.0, mutation_std=0.1, indpb=0.1)
        #self.update_rule_space.spaces["c1"].initialize(self)

        # UPDATE RULE
        ## resize R
        crea_parameters.R = crea['policy_parameters']['update_rule']['R'].to(device)
        crea_parameters.r = crea['policy_parameters']['update_rule']['r'].to(device) * scaling
        crea_parameters.T = crea['policy_parameters']['update_rule']['T'].to(device)


        crea_parameters.c0 = crea['policy_parameters']['update_rule']['c0'].to(device)
        crea_parameters.c1 = crea['policy_parameters']['update_rule']['c1'].to(device)
        crea_parameters.b = crea['policy_parameters']['update_rule']['b'].to(device)
        crea_parameters.m = crea['policy_parameters']['update_rule']['m'].to(device)
        crea_parameters.s = crea['policy_parameters']['update_rule']['s'].to(device)
        crea_parameters.h = crea['policy_parameters']['update_rule']['h'].to(device)

        if 'kn' not in crea['policy_parameters']['update_rule'] and 'w' in crea['policy_parameters']['update_rule']: # gautier version
            crea_parameters.kn = 4*torch.ones(nb_k, dtype=torch.int64)
            crea_parameters.gn = 1
            crea_parameters.w = crea['policy_parameters']['update_rule']['w'].to(device)
            crea_parameters.rk = crea['policy_parameters']['update_rule']['rk'].to(device)

        else:
            crea_parameters.kn = crea['policy_parameters']['update_rule']['kn']
            crea_parameters.gn = crea['policy_parameters']['update_rule']['gn'][0].item()
            crea_parameters.w=torch.ones((nb_k,3))
            crea_parameters.rk=torch.zeros((nb_k,3))



        self.reset(crea_parameters)

    def step(self, action=None, update_mask_rate=1.0, update_noise_rate=0.0, update_noise_std=0.0, obstacle_speed=0.0):
        if self.step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        # obstacle speed
        if self.config.wall_c and obstacle_speed > 0.0:
            if obstacle_speed < 1:
                shift = int(self.step_idx % int(1 / obstacle_speed) == 0)
            else:
                shift = int(obstacle_speed)
            self.state[0, :, :, -1] = torch.roll(self.state[0, :, :, -1], -shift, dims=-1)

        _ = self.lenia_step(self.state)

        # asynchronous updates
        if update_mask_rate < 1.0:
            rand_mask = (torch.rand(self.config.size) < update_mask_rate).type(self.state.dtype).to(self.state.device)
            self.lenia_step.D[0,:,:,0] *= rand_mask

        # noisy updates
        if update_noise_rate > 0.0:
            noise_mask = (torch.rand(self.config.size) < update_noise_rate).type(self.state.dtype).to(self.state.device)
            if update_noise_std > 0.0:
                noise_values = update_noise_std * torch.randn(self.config.size).type(self.state.dtype).to(self.state.device)

                self.lenia_step.D[0,:,:,0] = self.lenia_step.D[0,:,:,0] + noise_mask * noise_values

        self.state = torch.clamp(self.state + (1.0 / self.lenia_step.T) * self.lenia_step.D, min=0., max=1.)
        self.step_idx += 1
        
        self._observations.states[self.step_idx] = self.state[0]

        current_observation = Dict()
        current_observation.state = self._observations.states[self.step_idx]

        return current_observation, 0, self.step_idx >= self.config.final_step - 1, None

    def run(self, update_mask_rate=1.0, update_noise_rate=0.0, update_noise_std=0.0, obstacle_speed=0.0, perturb_until_step=None):
        if perturb_until_step is None:
            perturb_until_step = self.config.final_step - 1
        for step_idx in range(perturb_until_step):
            step_update_mask_rate = update_mask_rate
            while step_update_mask_rate > 1.0:
                self.step(update_mask_rate=1.0, update_noise_rate=update_noise_rate, update_noise_std=update_noise_std,
                          obstacle_speed=obstacle_speed)
                self.step_idx -= 1
                step_update_mask_rate = step_update_mask_rate - 1.0
            self.step(update_mask_rate=step_update_mask_rate, update_noise_rate=update_noise_rate, update_noise_std=update_noise_std,
                      obstacle_speed=obstacle_speed)
        for step_idx in range(perturb_until_step, self.config.final_step - 1):
            self.step()
        return self._observations



    def is_robust(self):
        return bool((self._observations.states[-1, 0, :, :].sum() > 10) and (
                self._observations.states[-1, 0, :, :].sum() < self._observations.states[3, 0, :, :].sum() * 4))

    def is_robust_2(self):
        return bool((self._observations.states[-1, 0, :, :].sum() > 10) and (
                self._observations.states[-1, 0, :, :].sum() < self._observations.states[0, 0, :, :].sum() * 3))

    def render(self, filename, frame_ids=None, size=None, crop=None, interpolation=Image.BILINEAR, fps=30.0,
               channel_colors=None, add_grid=None, acc_skip=27, mode='video', format='mp4'):
        if frame_ids is None:
            frame_ids = list(range(self.config.final_step))

        if channel_colors is None:
            channel_colors = [colors.to_rgb(color) for color in colors.TABLEAU_COLORS.values()][
                             :self.config.C ]
            channel_colors = np.array(channel_colors).transpose()

        # convert to numpy
        im_array = []
        for frame_idx in frame_ids:
            img = self._observations.states[frame_idx]
            slices = [slice(0, self.config.C )]
            for i, s in enumerate(self.config.size):
                if i < 2:
                    slices.append(slice(0, s))
                else:
                    slices.append(s // 2)
            img = channel_colors[:, :self.config.C ] @ img[slices].squeeze().cpu().detach().numpy().reshape(
                self.config.C, -1)
            im_array.append(img)

        if mode == "acc_img":
            acc_img = np.zeros_like(im_array[0])
            for timestep in range(len(im_array) // acc_skip):
                img = im_array[timestep * acc_skip]
                if ((timestep + 1) * acc_skip > len(im_array) - acc_skip):
                    acc_img += img
                else:
                    alpha = 1. * (((timestep + 1) * acc_skip) / (len(im_array) // acc_skip * acc_skip)) ** (0.4)
                    acc_img += alpha * img
            for c in range(acc_img.shape[0]):
                acc_img[c] = (acc_img[c] - acc_img[c].min()) / (acc_img[c].max() - acc_img[c].min())
            # acc_img = acc_img.clip(0,1)
            # acc_img = acc_img/acc_img.max()
            # plt.imshow(acc_img.reshape(3, self.config.size[0], self.config.size[1]).transpose(1,2,0))
            # plt.show()
            acc_img = np.uint8(255 * acc_img).reshape(3, self.config.size[0], self.config.size[1]).transpose(1, 2, 0)
            acc_im = Image.fromarray(acc_img, "RGB")
            acc_im.save(filename + "." + format)


        else:
            # convert to PIL
            for im_idx, img in enumerate(im_array):
                img = np.uint8(255 * img).reshape(3, self.config.size[0], self.config.size[1]).transpose(1, 2, 0)
                im = Image.fromarray(img, "RGB")
                if size is not None:
                    im = im.resize(size, interpolation)
                else:
                    size = (im.width, im.height)
                if add_grid:
                    draw = ImageDraw.Draw(im)
                    x_start = 0
                    x_end = im.width
                    y_start = 0
                    y_end = im.height
                    step_size = size[0] // self.config.size[0]
                    for x in range(0, im.width, step_size):
                        line = ((x, y_start), (x, y_end))
                        draw.line(line, fill=(255, 255, 255))
                    for y in range(0, im.height, step_size):
                        line = ((x_start, y), (x_end, y))
                        draw.line(line, fill=(255, 255, 255))
                if crop is not None:
                    im = im.crop(crop)
                im_array[im_idx] = im

            if mode == 'video':
                byte_img = io.BytesIO()
                imageio.mimwrite(byte_img, im_array, format=format, fps=fps, output_params=["-f", format])
                with open(filename + "." + format, "wb") as out_file:
                    out_file.write(byte_img.getbuffer())

            elif mode == 'img':
                for im_idx, im in enumerate(im_array):
                    im.save(filename + "_{}".format(im_idx) + "." + format)
                return im_array

        return

