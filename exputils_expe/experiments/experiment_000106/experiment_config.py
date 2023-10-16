from addict import Dict
from PIL import Image
import numpy as np
import os

save_observations = False
save_videos = True
save_images = False
seed = <repetition_id>

dset_name="imgep_exploration"

out_dir = os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/Figure_3/"+dset_name+"/seed"+str(seed)

def get_system_config():
    system_config = Dict()
    system_config.final_step = 2000
    system_config.nb_c = 1
    system_config.nb_k = 10
    system_config.SX = 256
    system_config.SY = 256
    system_config.wall_c = bool(0)

    return system_config


def get_stat_config():
    stat_config = Dict()

    stat_config.categories_filepath = os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/resources/creatures_categoriesV2.json"
    stat_config.categories_filter = Dict(is_robust=True, is_long_term_stable=True, is_soliton=True)

    stat_config.dset_name = dset_name
    stat_config.var_name = "scaling"

    stat_config.init_noise_rate = Dict()
    stat_config.init_noise_rate.range = list([])
    stat_config.init_noise_rate.reset_kwargs = Dict()
    stat_config.init_noise_rate.reset_kwargs.init_noise_rate = None
    stat_config.init_noise_rate.reset_kwargs.init_noise_std = 1.0
    stat_config.init_noise_rate.run_kwargs = Dict()

    stat_config.init_noise_std = Dict()
    stat_config.init_noise_std.range = list([])
    stat_config.init_noise_std.reset_kwargs = Dict()
    stat_config.init_noise_std.reset_kwargs.init_noise_std = None
    stat_config.init_noise_std.reset_kwargs.init_noise_rate = 1.0
    stat_config.init_noise_std.run_kwargs = Dict()


    stat_config.obstacle_number = Dict()
    stat_config.obstacle_number.range = list([])
    stat_config.obstacle_number.reset_kwargs = Dict()
    stat_config.obstacle_number.reset_kwargs.obstacle_number = None
    stat_config.obstacle_number.reset_kwargs.obstacle_radius = 10
    stat_config.obstacle_number.reset_kwargs.obstacle_protection = "blurr+10"
    stat_config.obstacle_number.run_kwargs = Dict()

    stat_config.obstacle_radius = Dict()
    stat_config.obstacle_radius.range = list([])
    stat_config.obstacle_radius.reset_kwargs = Dict()
    stat_config.obstacle_radius.base = Dict()
    stat_config.obstacle_radius.reset_kwargs.obstacle_radius = None
    stat_config.obstacle_radius.reset_kwargs.obstacle_number = 24
    stat_config.obstacle_radius.base.obstacle_number_base = 24
    stat_config.obstacle_radius.reset_kwargs.obstacle_protection = "blurr+10"
    stat_config.obstacle_radius.run_kwargs = Dict()

    stat_config.obstacle_speed = Dict()
    stat_config.obstacle_speed.range = list([])
    stat_config.obstacle_speed.reset_kwargs = Dict()
    stat_config.obstacle_speed.reset_kwargs.obstacle_radius = 10
    stat_config.obstacle_speed.reset_kwargs.obstacle_number = 24
    stat_config.obstacle_speed.reset_kwargs.obstacle_protection = "blurr+10"
    stat_config.obstacle_speed.run_kwargs = Dict()
    stat_config.obstacle_speed.run_kwargs.obstacle_speed = None

    stat_config.scaling = Dict()
    stat_config.scaling.range = np.linspace(0.15,2.15,5)
    stat_config.scaling.reset_kwargs = Dict()
    stat_config.scaling.reset_kwargs.scaling = None
    stat_config.scaling.run_kwargs = Dict()

    stat_config.update_mask_rate = Dict()
    stat_config.update_mask_rate.range = list([])
    stat_config.update_mask_rate.reset_kwargs = Dict()
    stat_config.update_mask_rate.run_kwargs = Dict()
    stat_config.update_mask_rate.run_kwargs.update_mask_rate = None

    stat_config.update_noise_rate = Dict()
    stat_config.update_noise_rate.range = list([])
    stat_config.update_noise_rate.reset_kwargs = Dict()
    stat_config.update_noise_rate.run_kwargs = Dict()
    stat_config.update_noise_rate.run_kwargs.update_noise_rate = None
    stat_config.update_noise_rate.run_kwargs.update_noise_std = 1.0
    stat_config.update_noise_rate.run_kwargs.perturb_until_step = 1900 - 1

    stat_config.update_noise_std = Dict()
    stat_config.update_noise_std.range = list([])
    stat_config.update_noise_std.reset_kwargs = Dict()
    stat_config.update_noise_std.run_kwargs = Dict()
    stat_config.update_noise_std.run_kwargs.update_noise_std = None
    stat_config.update_noise_std.run_kwargs.update_noise_rate = 1.0
    stat_config.update_noise_std.run_kwargs.perturb_until_step = 1900 - 1


    return stat_config

def get_rendering_config():
    rendering_config = Dict()
    rendering_config.size = (256, 256)
    rendering_config.interpolation = Image.BILINEAR
    rendering_config.channel_colors = np.array([[1.0,1.0,0.0],[0.0,0.0,1.0]]).transpose()

    rendering_config.fps = 120

    return rendering_config
