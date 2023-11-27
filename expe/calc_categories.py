import numpy as np
import os
import json
import torch

""" --------------------------------------------------------------------------------------------------------
FILTERS
-------------------------------------------------------------------------------------------------------- """


# def is_animal(observations, r=1, tol=0.1, obs1_idx=-1, obs2_idx=-2, min_activity=0.8):
#     animal_segments_obs1 = []
#     animal_segments_obs2 = []
#
#     obs1 = observations[obs1_idx]
#
#     finite_segments_obs1, (segmented_image_obs1, _) = calc_is_segments_finite(obs1, tol=tol, r=r)
#
#     if not finite_segments_obs1:
#         return False
#
#     else:
#
#         activity = np.sum(obs1)
#
#         for seg_idx in finite_segments_obs1:
#             cur_activity = np.sum(obs1[segmented_image_obs1 == seg_idx])
#
#             if cur_activity / activity >= min_activity:
#                 animal_segments_obs1.append(seg_idx)
#
#         if not animal_segments_obs1:
#             return False
#
#         else:
#             obs2 = observations[obs2_idx]
#
#             finite_segments_obs2, (segmented_image_obs2, _) = calc_is_segments_finite(obs2, tol=tol, r=r)
#
#             activity = np.sum(obs2)
#
#             for seg_idx in finite_segments_obs2:
#                 cur_activity = np.sum(obs2[segmented_image_obs2 == seg_idx])
#
#                 if cur_activity / activity >= min_activity:
#                     animal_segments_obs2.append(seg_idx)
#
#             if not animal_segments_obs2:
#                 return False
#
#     return True


def is_robust(statistics, low_mass_threshold=0, high_mass_threshold=6400, num_cells=65536):
    final_mass = statistics['activation_mass'][-1].item() * num_cells
    return bool((final_mass > low_mass_threshold) and (final_mass < high_mass_threshold))


def is_long_term_stable(statistics, phase1_timepoints=(1/4, 1/2), phase2_timepoints=(3/4, 1), low_ratio_threshold=0.5, high_ratio_threshold=3.0):
    activation_mass = statistics['activation_mass']
    n_steps = activation_mass.shape[0]

    phase1_start_idx = int(n_steps * phase1_timepoints[0])
    phase1_end_idx = int(n_steps * phase1_timepoints[1])

    phase2_start_idx = int(n_steps * phase2_timepoints[0])
    phase2_end_idx = int(n_steps * phase2_timepoints[1])

    phase1_avg = np.nanmean(activation_mass[phase1_start_idx:phase1_end_idx])
    phase2_avg = np.nanmean(activation_mass[phase2_start_idx:phase2_end_idx])

    ratio = phase2_avg / phase1_avg

    return bool((low_ratio_threshold < ratio) and (ratio < high_ratio_threshold))


def is_soliton(statistics, low_nr_objects=0, high_nr_objects=2, max_activity=256*256*0.6):
    activity = statistics["connected_components_activity"]
    nr_objects = statistics["connected_components_nr_objects"]
    return bool((low_nr_objects < nr_objects < high_nr_objects) and activity.sum() < max_activity)

def is_moving(statistics, min_distance_from_init=100, final_step=500):
    center = statistics['activation_center_position'][:final_step, :]
    return bool(np.linalg.norm(center-center[0], ord=2, axis=-1).max() > min_distance_from_init)



if __name__ == "__main__":

    json_data = {}
    json_filepath = '../data/creatures_categories.json'
    resources_dir = "../data/"
    dset_directories = ['imgep_exploration', 'random_exploration', 'handmade_exploration']
    
    
    
    for dset_dir in dset_directories:
        json_data[dset_dir] = {}
        for crea_filename in os.listdir(resources_dir+dset_dir+"/prefilter_parameters"):
            if crea_filename[-7:] == ".pickle":
                crea_name = crea_filename[:-7]
                if crea_name not in json_data[dset_dir]:
                    json_data[dset_dir][crea_name] = {}
                    json_data[dset_dir][crea_name]['parameters'] = resources_dir + dset_dir+"/prefilter_parameters/"+crea_filename
                    assert os.path.exists(json_data[dset_dir][crea_name]['parameters'])
                    json_data[dset_dir][crea_name]['observations'] = resources_dir + dset_dir+"/observations/observations_"+crea_filename
                    assert os.path.exists(json_data[dset_dir][crea_name]['observations'])
                    json_data[dset_dir][crea_name]['stats'] = resources_dir + dset_dir+"/stats/stats_"+crea_filename
                    assert os.path.exists(json_data[dset_dir][crea_name]['stats'])
                    json_data[dset_dir][crea_name]['video'] = resources_dir + dset_dir + "/videos/" + crea_name + ".mp4"


                    statistics = torch.load(json_data[dset_dir][crea_name]['stats'])
                    final_step = statistics['activation_mass'].shape[0]

                    json_data[dset_dir][crea_name]['is_robust'] = bool(
                        is_robust(statistics, low_mass_threshold=0, high_mass_threshold=6400, num_cells=256*256))

                    json_data[dset_dir][crea_name]['is_long_term_stable'] = bool(is_long_term_stable(statistics,
                                                                         phase1_timepoints=(0, 1 / 4),
                                                                         phase2_timepoints=(3 / 4, 1),
                                                                         low_ratio_threshold=0.,
                                                                         high_ratio_threshold=2.0))
                    json_data[dset_dir][crea_name]['is_soliton'] = bool(
                        is_soliton(statistics, low_nr_objects=0, high_nr_objects=2, max_activity=0.6*256*256))

                    json_data[dset_dir][crea_name]['is_moving'] = bool(
                        is_moving(statistics, min_distance_from_init=100, final_step=1000))

    with open(json_filepath, 'w') as out_file:
        json.dump(json_data, out_file)
