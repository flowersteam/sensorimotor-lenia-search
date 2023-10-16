from experiment_config import *
import torch
import numpy as np
import os
import cmath
import collections
import cv2
import json
import math
from scipy import ndimage
        
torch.manual_seed(seed)
np.random.seed(seed)


""" --------------------------------------------------------------------------------------------------------
STATISTICS 
-------------------------------------------------------------------------------------------------------- """




def calc_center_of_mass(image):
    center = np.array(ndimage.measurements.center_of_mass(image))

    if np.any(np.isnan(center)):
        center = np.array([int((image.shape[0] - 1) / 2), int((image.shape[1] - 1) / 2)])

    return center


def calc_image_moments(image):
    '''
    Calculates the image moments for an image.

    For more information see:
     - https://en.wikipedia.org/wiki/Image_moment
     - http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/SHUTLER3/node1.html

    The code is based on the Javascript implementation of Lenia by Bert Chan.
    The code had to be adapted, because for the original code the coordinates are (x,y), whereas here they are (y,x).

    :param image: 2d gray scale image in form of a numpy array.
    :return: Namedtupel with the different moments.
    '''

    eps = 0.00001
    Moments = collections.namedtuple('Moments', ['y_avg', 'x_avg',
                                                 'm00', 'm10', 'm01', 'm11', 'm20', 'm02', 'm21', 'm12', 'm22', 'm30',
                                                 'm31', 'm13', 'm03', 'm40', 'm04',
                                                 'mu11', 'mu20', 'mu02', 'mu30', 'mu03', 'mu21', 'mu12', 'mu22', 'mu31',
                                                 'mu13', 'mu40', 'mu04',
                                                 'eta11', 'eta20', 'eta02', 'eta30', 'eta03', 'eta21', 'eta12', 'eta22',
                                                 'eta31', 'eta13', 'eta40', 'eta04',
                                                 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 'hu8',
                                                 'flusser9', 'flusser10', 'flusser11', 'flusser12', 'flusser13'])

    size_y = image.shape[0]
    size_x = image.shape[1]

    x_grid, y_grid = np.meshgrid(range(size_x), range(size_y))

    y_power1_image = y_grid * image
    y_power2_image = y_grid * y_power1_image
    y_power3_image = y_grid * y_power2_image

    x_power1_image = x_grid * image
    x_power2_image = x_grid * x_power1_image
    x_power3_image = x_grid * x_power2_image

    # raw moments: m_qp
    m00 = np.sum(image)
    m10 = np.sum(y_power1_image)
    m01 = np.sum(x_power1_image)
    m11 = np.sum(y_grid * x_grid * image)
    m20 = np.sum(y_power2_image)
    m02 = np.sum(x_power2_image)
    m21 = np.sum(y_power2_image * x_grid)
    m12 = np.sum(x_power2_image * y_grid)
    m22 = np.sum(x_power2_image * y_grid * y_grid)
    m30 = np.sum(y_power3_image)
    m31 = np.sum(y_power3_image * x_grid)
    m13 = np.sum(y_grid * x_power3_image)
    m03 = np.sum(x_power3_image)
    m40 = np.sum(y_power3_image * y_grid)
    m04 = np.sum(x_power3_image * x_grid)

    # mY and mX describe the position of the centroid of the image
    # if there is no activation, then use the center position
    if m00 == 0:
        mY = (image.shape[0] - 1) / 2
        mX = (image.shape[1] - 1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00

    # in the case of very small activation (m00 ~ 0) the position becomes infinity, also then use the center position
    if mY == float('inf'):
        mY = (image.shape[0] - 1) / 2
    if mX == float('inf'):
        mX = (image.shape[1] - 1) / 2

    # calculate the central moments
    X2 = mX * mX
    X3 = X2 * mX
    Y2 = mY * mY
    Y3 = Y2 * mY
    XY = mX * mY

    mu11 = m11 - mY * m01
    mu20 = m20 - mY * m10
    mu02 = m02 - mX * m01
    mu30 = m30 - 3 * mY * m20 + 2 * Y2 * m10
    mu03 = m03 - 3 * mX * m02 + 2 * X2 * m01
    mu21 = m21 - 2 * mY * m11 - mX * m20 + 2 * Y2 * m01
    mu12 = m12 - 2 * mX * m11 - mY * m02 + 2 * X2 * m10
    mu22 = m22 - 2 * mX * m21 + X2 * m20 - 2 * mY * m12 + 4 * XY * m11 - 2 * mY * X2 * m10 + Y2 * m02 - 2 * Y2 * mX * m01 + Y2 * X2 * m00
    mu31 = m31 - mX * m30 + 3 * mY * (mX * m20 - m21) + 3 * Y2 * (m11 - mX * m10) + Y3 * (mX * m00 - m01)
    mu13 = m13 - mY * m03 + 3 * mX * (mY * m02 - m12) + 3 * X2 * (m11 - mY * m01) + X3 * (mY * m00 - m10)
    mu40 = m40 - 4 * mY * m30 + 6 * Y2 * m20 - 4 * Y3 * m10 + Y2 * Y2 * m00
    mu04 = m04 - 4 * mX * m03 + 6 * X2 * m02 - 4 * X3 * m01 + X2 * X2 * m00

    # Moment invariants: scale invariant
    if m00 < eps:
        eta11 = 0
        eta20 = 0
        eta02 = 0
        eta30 = 0
        eta03 = 0
        eta21 = 0
        eta12 = 0
        eta22 = 0
        eta31 = 0
        eta13 = 0
        eta40 = 0
        eta04 = 0
    else:
        m2 = m00 * m00
        mA = m00 * m00 * math.sqrt(m00)
        m3 = m00 * m00 * m00
        eta11 = mu11 / m2
        eta20 = mu20 / m2
        eta02 = mu02 / m2
        eta30 = mu30 / mA
        eta03 = mu03 / mA
        eta21 = mu21 / mA
        eta12 = mu12 / mA
        eta22 = mu22 / m3
        eta31 = mu31 / m3
        eta13 = mu13 / m3
        eta40 = mu40 / m3
        eta04 = mu04 / m3

    # Moment invariants: rotation invariants
    Z = 2 * eta11
    A = eta20 + eta02
    B = eta20 - eta02
    C = eta30 + eta12
    D = eta30 - eta12
    E = eta03 + eta21
    F = eta03 - eta21
    G = eta30 - 3 * eta12
    H = 3 * eta21 - eta03
    Y = 2 * eta22
    I = eta40 + eta04
    J = eta40 - eta04
    K = eta31 + eta13
    L = eta31 - eta13
    CC = C * C
    EE = E * E
    CC_EE = CC - EE
    CC_EE3 = CC - 3 * EE
    CC3_EE = 3 * CC - EE
    CE = C * E
    DF = D * F
    M = I - 3 * Y
    t1 = CC_EE * CC_EE - 4 * CE * CE
    t2 = 4 * CE * CC_EE

    # invariants by Hu
    hu1 = A
    hu2 = B * B + Z * Z
    hu3 = G * G + H * H
    hu4 = CC + EE
    hu5 = G * C * CC_EE3 + H * E * CC3_EE
    hu6 = B * CC_EE + 2 * Z * CE
    hu7 = H * C * CC_EE3 - G * E * CC3_EE
    hu8 = Z * CC_EE / 2 - B * CE

    # extra invariants by Flusser
    flusser9 = I + Y
    flusser10 = J * CC_EE + 4 * L * DF
    flusser11 = -2 * K * CC_EE - 2 * J * DF
    flusser12 = 4 * L * t2 + M * t1
    flusser13 = -4 * L * t1 + M * t2

    result = Moments(y_avg=mY, x_avg=mX,
                     m00=m00, m10=m10, m01=m01, m11=m11, m20=m20, m02=m02, m21=m21, m12=m12, m22=m22, m30=m30, m31=m31,
                     m13=m13, m03=m03, m40=m40, m04=m04,
                     mu11=mu11, mu20=mu20, mu02=mu02, mu30=mu30, mu03=mu03, mu21=mu21, mu12=mu12, mu22=mu22, mu31=mu31,
                     mu13=mu13, mu40=mu40, mu04=mu04,
                     eta11=eta11, eta20=eta20, eta02=eta02, eta30=eta30, eta03=eta03, eta21=eta21, eta12=eta12,
                     eta22=eta22, eta31=eta31, eta13=eta13, eta40=eta40, eta04=eta04,
                     hu1=hu1, hu2=hu2, hu3=hu3, hu4=hu4, hu5=hu5, hu6=hu6, hu7=hu7, hu8=hu8,
                     flusser9=flusser9, flusser10=flusser10, flusser11=flusser11, flusser12=flusser12,
                     flusser13=flusser13)

    return result


def angle_of_vec_degree(vector):
    # if vector[0] >= 0:
    #     return np.arctan2(vector[0], vector[1]) * 180/np.pi
    # else:
    #     return 360+(np.arctan2(vector[0], vector[1]) * 180 / np.pi)

    if vector[1] >= 0:
        return np.arctan2(vector[1], vector[0]) * 180 / np.pi
    else:
        return 360 + (np.arctan2(vector[1], vector[0]) * 180 / np.pi)


def angle_difference_degree(angle1, angle2):
    '''
    Calculates the disctance between two angles which are in degree.

    If the distance is clockwise, its negative, otherwise positive.

    :param angle1: First angle. Either scalar or array.
    :param angle2: Second angle. Either scalar or array.
    :return: Distance between angles in degrees.
    '''
    if isinstance(angle1, list):
        angle1 = np.array(angle1)

    if isinstance(angle2, list):
        angle2 = np.array(angle2)

    phi = np.mod(angle2 - angle1, 360)

    if not isinstance(phi, np.ndarray):
        sign = 1
        # used to calculate sign
        if not ((phi >= 0 and phi <= 180) or (
                phi <= -180 and phi >= -360)):
            sign = -1
        if phi > 180:
            result = 360 - phi
        else:
            result = phi

    else:
        sign = np.ones(phi.shape)

        sign[np.logical_not(
            np.logical_or(np.logical_and(phi >= 0, phi <= 180), np.logical_and(phi <= -180, phi >= -360)))] = -1

        result = phi
        result[phi > 180] = 360 - phi[phi > 180]

    return result * sign


def mean_over_angles_degrees(angles):
    '''Calculates the mean over angles that are given in degrees.'''
    if len(angles) == 0:
        return np.nan
    else:
        return math.degrees(cmath.phase(sum(cmath.rect(1, math.radians(d)) for d in angles) / len(angles)))


def nan_mean_over_angles_degrees(angles):
    '''Calculates the mean over angles that are given in degrees. Ignores nan values.'''
    np_angles = np.array(angles)
    return mean_over_angles_degrees(np_angles[~np.isnan(np_angles)])



def calc_connected_components(image, blur_radius=10, threshold_prior_blur=0.05, threshold_post_blur=0.05):

    # smooth the image to connect region in radius of influence
    S=int(2*blur_radius)
    x = np.arange(S)
    y = np.arange(S)
    xx = x.reshape(-1, 1).repeat(S, 1)
    yy = y.reshape(1, -1).repeat(S, 0)
    X = (xx - (S-1)/2)
    Y = (yy - (S-1)/2)
    D = np.sqrt(X ** 2 + Y ** 2) / blur_radius
    kernel = (D < 1).astype(image.dtype)
    imgf = cv2.filter2D((image > threshold_prior_blur).astype(image.dtype), -1, kernel)
    # find connected components
    labeled, nr_objects = ndimage.label((imgf > threshold_post_blur).astype(image.dtype))

    return labeled*(image > threshold_prior_blur).astype(labeled.dtype), nr_objects



def calc_statistics(all_obs, parameters):
    num_of_obs = len(all_obs)
    activation_mass_data = np.ones(num_of_obs) * np.nan
    activation_center_position_data = np.ones((num_of_obs, 2)) * np.nan
    activation_center_velocity_data = np.ones(num_of_obs) * np.nan
    activation_center_movement_angle_data = np.ones(num_of_obs) * np.nan
    activation_center_movement_angle_velocity_data = np.ones(num_of_obs) * np.nan

    size_y = all_obs[0].shape[0]
    size_x = all_obs[0].shape[1]
    num_of_cells = size_y * size_x

    # calc initial center of mass and use it as a reference point to "center" the world around it
    # in consequetive steps, recalculate the center of mass and "recenter" the wolrd around them
    mid_y = (size_y - 1) / 2
    mid_x = (size_x - 1) / 2
    mid = np.array([mid_y, mid_x])

    activation_center_of_mass = np.array(calc_center_of_mass(all_obs[0]))
    activation_shift_to_center = mid - activation_center_of_mass
    prev_activation_center_movement_angle = np.nan
    uncentered_activation_center_position = np.array([np.nan, np.nan])

    for step in range(num_of_obs):
        activation = all_obs[step]

        # shift the system to the last calculated center of mass so that it is in the middle
        # the matrix can only be shifted in discrete values, therefore the shift is transformed to integer
        centered_activation = np.roll(activation, activation_shift_to_center.astype(int), (0, 1))

        # calculate the image moments
        activation_moments = calc_image_moments(centered_activation)

        # new center of mass
        activation_center_of_mass = np.array([activation_moments.y_avg, activation_moments.x_avg])

        # calculate the change of center as a vector
        activation_shift_from_prev_center = mid - activation_center_of_mass

        # calculate the new shift to center the next obs to the new center
        activation_shift_to_center = activation_shift_to_center.astype(int) + activation_shift_from_prev_center

        # transform the new center, encoded as a shift from the first image, back into the original image coordinates

        activation_center_position_data[step] = activation_shift_to_center*1.0

        # activation mass
        activation_mass = activation_moments.m00
        activation_mass_data[step] = activation_mass / num_of_cells  # activation is number of acitvated cells divided by the number of cells

        # get velocity and angle of movement
        #   distance between the previous center of mass and the new one is the velocity
        #   angle is computed based on the shift vector
        if step <= 0:
            activation_center_velocity = np.nan
            activation_center_movement_angle = np.nan
            activation_center_movement_angle_velocity = np.nan
        else:
            activation_center_velocity = np.linalg.norm(activation_shift_from_prev_center)

            if activation_center_velocity == 0:
                activation_center_movement_angle = np.nan
            else:
                activation_center_movement_angle = angle_of_vec_degree(
                    [-1 * activation_shift_from_prev_center[1], activation_shift_from_prev_center[0]])

            # Angular velocity, is the difference between the current and previous angle of movement
            if activation_center_movement_angle is np.nan or prev_activation_center_movement_angle is np.nan:
                activation_center_movement_angle_velocity = 0
            else:
                activation_center_movement_angle_velocity = angle_difference_degree(activation_center_movement_angle,
                                                                                    prev_activation_center_movement_angle)

            prev_activation_center_movement_angle = activation_center_movement_angle

        activation_center_velocity_data[step] = activation_center_velocity
        activation_center_movement_angle_data[step] = activation_center_movement_angle
        activation_center_movement_angle_velocity_data[step] = activation_center_movement_angle_velocity


    statistics = dict()
    statistics['activation_mass'] = activation_mass_data
    statistics['activation_center_position'] = activation_center_position_data
    statistics['activation_center_velocity'] = activation_center_velocity_data
    statistics['activation_center_movement_angle'] = activation_center_movement_angle_data
    statistics['activation_center_movement_angle_velocity'] = activation_center_movement_angle_velocity_data
    R = parameters['policy_parameters']['update_rule']['R'].item() + 15
    r = parameters['policy_parameters']['update_rule']['r']
    blur_radius = R * r.max().item() / 2.0
    label, nr_objects = calc_connected_components(centered_activation, blur_radius=blur_radius)
    statistics['connected_components_activity'] = (label > 0).sum()
    statistics['connected_components_nr_objects'] = nr_objects

    return statistics


""" --------------------------------------------------------------------------------------------------------
FILTERS
-------------------------------------------------------------------------------------------------------- """



def is_robust(statistics, low_mass_threshold=0, high_mass_threshold=6400):
    final_mass = statistics['activation_mass'][-1].sum()
    return bool(( final_mass> low_mass_threshold) and (final_mass < high_mass_threshold))


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


def is_soliton(statistics, low_nr_objects=0, high_nr_objects=2):
    return bool(low_nr_objects < statistics["connected_components_nr_objects"] < high_nr_objects)


def is_moving(statistics, min_distance_from_mean_pos=10, windows=[slice(-50,None)], n_active_windows=1):
    # see if there is a change of the center position in the last x time steps
    activation_center_velocity = statistics['activation_center_velocity']
    activation_center_movement_angle = statistics['activation_center_movement_angle']

    is_moving = 0
    for window in windows:

        # calculate the relative position of the activation center beginning fom the last n step
        # we are not using the activation_center statistic, because it changes drastically if the center is near the image border

        # get shifts of each step
        movement_angles_in_rad = np.radians(activation_center_movement_angle[window])
        movement_distance = activation_center_velocity[window]
        x_shift = np.cos(movement_angles_in_rad) * movement_distance
        y_shift = np.sin(movement_angles_in_rad) * movement_distance
        x_shift[movement_distance == 0] = 0
        y_shift[movement_distance == 0] = 0

        # get relative positions
        relative_x_pos = np.cumsum(x_shift)
        relative_y_pos = np.cumsum(y_shift)
        relative_pos = np.vstack((relative_x_pos, relative_y_pos)).transpose()

        # relative mean position in last n steps
        mean_pos = np.nanmean(relative_pos, axis=0)

        # calc dist of each position to the mean position
        dist_to_mean_pos = np.linalg.norm(mean_pos - relative_pos, axis=1)

        # detect movement if the distance of one pos is larger than x to the mean pos
        is_moving += int(np.nanmax(dist_to_mean_pos) > min_distance_from_mean_pos)

    return is_moving >= n_active_windows


def get_categories(crea_filepaths, out_stat_filepath, init_hflip=True, final_step=1000,
                   nb_k=10, size=(256, 256)):
    system = TmpPytorchLenia(logger=None, final_step=final_step, nb_c=1,
                             nb_k=nb_k, wall_c=False,
                             size="({},{})".format(size[0], size[1]))

    with open(out_stat_filepath, 'w') as out_stat_file:
        out_stat_data = {}
        json.dump(out_stat_data, out_stat_file)

    for crea_name, crea_filepath in crea_filepaths.items():
        parameters = torch.load(crea_filepath[:-14] + '_params.pickle')

        if not os.path.exists(crea_filepath[:-14] + '_states.pickle'):
            system.reset_from_crea_filepath(crea_filepath, init_hflip=init_hflip)
            system.run()
            # save rollout states
            with open(crea_filepath[:-14] + '_states.pickle', 'wb') as out_file:
                torch.save(system._observations.states, out_file)

            # save rollout video
            system.render(frame_ids=range(system.config.final_step - 1), filename=crea_filepath[:-14],
                          mode='mp4', channel_colors=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).transpose())

            observations = system._observations.states

        else:
            observations = torch.load(crea_filepath[:-14] + '_states.pickle')\

        observations = observations[:,0,:,:].cpu().numpy()

        if not os.path.exists(crea_filepath[:-14] + '_stats.npy'):
            statistics = calc_statistics(observations, parameters)

            # save statistics
            with open(crea_filepath[:-14] + '_stats.npy', 'wb') as out_file:
                np.savez(out_file, **statistics)

        else:
            statistics = np.load(crea_filepath[:-14] + '_stats.npy')


        with open(out_stat_filepath, 'r+') as out_stat_file:
            out_stat_data = json.load(out_stat_file)
            out_stat_data[crea_name] = {}
            out_stat_data[crea_name]['is_robust'] = bool(is_robust(observations, low_mass_threshold=0, high_mass_threshold=6400))
            out_stat_data[crea_name]['is_long_term_stable'] = bool(is_long_term_stable(statistics, low_ratio_threshold=0.5, high_ratio_threshold=3.0))
            out_stat_data[crea_name]['is_soliton'] = bool(is_soliton(statistics), phase1_timepoints=(1/4, 1/2), phase2_timepoints=(3/4, 1), low_ratio_threshold=0.5, high_ratio_threshold=3.0)
            out_stat_data[crea_name]['is_moving'] = bool(is_moving(statistics, min_distance_from_mean_pos=3, windows=[slice(t_start, t_end) for t_start, t_end in zip(range(0, final_step-50+1, 25), range(50, final_step+1, 25))], n_active_windows=10))


        with open(out_stat_filepath, 'w') as out_stat_file:
            json.dump(out_stat_data, out_stat_file, indent=4)


if __name__ == "__main__":
    pf=os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/resources/"+type_expe+"_exploration/"


    path=pf+"parameters"

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path,file)) and file.startswith('seed'+str(seed)):


            observations=torch.load(pf+"observations/observations_"+file,map_location='cpu')
            parameters = torch.load(os.path.join(path,file),map_location='cpu')
            
            
            statistics = calc_statistics(observations[:,:,:,0].numpy(), parameters)
                
            torch.save(statistics,pf+"stats/stats_"+file)
            
            print(file,statistics['connected_components_nr_objects'])
            


        

print("finished")

