from addict import Dict
import json
import numpy as np
import os
from sensorimotorleniasearch import TmpLenia
from sensorimotorleniasearch import calc_statistics
import torch
import random

import experiment_config


def _set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def calc_robustness_to_env_var():

    # Prepare Experiment
    out_folder = f"{experiment_config.out_dir}"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    stats_folder = f"{out_folder}/stats"
    if not os.path.isdir(stats_folder):
        os.makedirs(stats_folder)
    if experiment_config.save_observations:
        observations_folder = f"{out_folder}/observations"
        if not os.path.isdir(observations_folder):
            os.makedirs(observations_folder)
    if experiment_config.save_videos:
        videos_folder = f"{out_folder}/videos"
        if not os.path.isdir(videos_folder):
            os.makedirs(videos_folder)
    if experiment_config.save_images:
        images_folder = f"{out_folder}/images"
        if not os.path.isdir(images_folder):
            os.makedirs(images_folder)

    system_config = experiment_config.get_system_config()

    stat_config = experiment_config.get_stat_config()

    rendering_config = experiment_config.get_rendering_config()



    # Prepare System
    system = TmpLenia(logger=None, final_step=system_config.final_step, nb_c=system_config.nb_c,
                             nb_k=system_config.nb_k, wall_c=system_config.wall_c,
                             size="({},{})".format(system_config.SX, system_config.SY))


    # Prepare Dataset
    with open(stat_config.categories_filepath, 'r') as categories_file:
        categories_data = json.load(categories_file)
    dataset = categories_data[stat_config.dset_name]
    filtered_dataset = Dict()
    for crea_name in dataset.keys():
        is_valid = True
        for k, v in stat_config.categories_filter.items():
            if dataset[crea_name][k] != v:
                is_valid = False
                break
        if is_valid:
            filtered_dataset[crea_name] = dataset[crea_name]


    # Run test experiment over dataset
    var_name = stat_config.var_name
    for var in stat_config[var_name].range:
        if var_name in stat_config[var_name].reset_kwargs.keys():
            stat_config[var_name].reset_kwargs[var_name] = var
            if var_name == "obstacle_radius":
                stat_config[var_name].reset_kwargs["obstacle_number"] = int(stat_config[var_name].base["obstacle_number_base"] * (10/var)**2)
        elif var_name in stat_config[var_name].run_kwargs.keys():
            stat_config[var_name].run_kwargs[var_name] = var


        for crea_idx, crea_name in enumerate(list(filtered_dataset.keys())):
            print(crea_name)
            _set_seed(experiment_config.seed*10000+crea_idx)

            out_filename = f"{crea_name}_{var_name}_var_{var:.2f}"

            # save statistics
            out_stats_filepath = f"{stats_folder}/stats_{out_filename}.pickle"
            is_stat_valid = True
            if os.path.exists(out_stats_filepath):
                try:
                    statistics = torch.load(out_stats_filepath)
                except:
                    is_stat_valid = False

            if not os.path.exists(out_stats_filepath) or not is_stat_valid:
                parameters = torch.load(filtered_dataset[crea_name]['parameters'], map_location='cpu')

                # LENIA ROLLOUT
                if "obstacle" in var_name:
                    stats = torch.load(filtered_dataset[crea_name]['stats'])
                    abs_pos = stats["activation_center_position"]
                    obstacle_pos = torch.from_numpy(np.mod(abs_pos[1000], 256)).unsqueeze(0)
                    stat_config[var_name].reset_kwargs["obstacle_positions"] = obstacle_pos


                try:
                    
                    system.reset_from_crea_filepath(filtered_dataset[crea_name]['parameters'],
                                                    **stat_config[var_name].reset_kwargs)
                    with torch.no_grad():
                        system.run(**stat_config[var_name].run_kwargs)
                    observations = system._observations.states
                    statistics = calc_statistics(observations[:, :, :, 0].cpu().numpy(), parameters)

                except:
                    observations = None
                    statistics = dict()
                    statistics['activation_mass'] = None
                    statistics['activation_center_position'] = None
                    statistics['activation_center_velocity'] = None
                    statistics['activation_center_movement_angle'] = None
                    statistics['activation_center_movement_angle_velocity'] = None
                    statistics['connected_components_activity'] = None
                    statistics['connected_components_nr_objects'] = None

                torch.save(statistics, out_stats_filepath)

                if experiment_config.save_observations:
                    out_observations_filepath = f"{observations_folder}/observations_{out_filename}.pickle"
                    torch.save(observations, out_observations_filepath)

                # save video
                if experiment_config.save_videos and observations is not None:
                    out_video_filepath = f"{videos_folder}/{out_filename}.mp4"
                    if not os.path.exists(out_video_filepath):
                        observations = observations.swapaxes(3, 1)
                        observations = observations.swapaxes(2, 3)
                        system._observations = Dict()
                        system._observations.states = observations

                        try:
                            system.render(frame_ids=range(system_config.final_step - 1),
                                          filename=out_video_filepath[:-4],
                                          size=rendering_config.size,
                                          interpolation=rendering_config.interpolation,
                                          channel_colors=rendering_config.channel_colors,
                                          fps=rendering_config.fps,
                                          mode='video',
                                          format='mp4')
                        except:
                            pass

                # save image
                if experiment_config.save_images and observations is not None:
                    out_image_filepath = f"{images_folder}/{out_filename}.pdf"
                    if not os.path.exists(out_image_filepath):
                        observations = observations.swapaxes(3, 1)
                        system._observations = Dict()
                        system._observations.states = observations

                        system.render(frame_ids=range(system_config.final_step - 1),
                                      filename=out_image_filepath[:-4],
                                      size=rendering_config.size,
                                      interpolation=rendering_config.interpolation,
                                      channel_colors=rendering_config.channel_colors,
                                      mode='acc_img',
                                      acc_skip=50,
                                      format='pdf')




if __name__ == '__main__':
    _set_seed(experiment_config.seed)
    calc_robustness_to_env_var()
