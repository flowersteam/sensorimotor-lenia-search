from experiment_config import *
import torch
import json
import os
from addict import Dict
from sensorimotorleniasearch.systems import Lenia_C
from sensorimotorleniasearch.systems import Lenia_C_move
import numpy as np
import torch
from sensorimotorleniasearch.systems.lenia import LeniaInitializationSpace
import cmath
import collections
import cv2
import math
from scipy import ndimage
import time


torch.manual_seed(seed)
np.random.seed(seed)


crea_seed=seed//10
sub_seed=seed%10


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
    label, nr_objects = ndimage.label((imgf > threshold_post_blur).astype(image.dtype))

    return label, nr_objects



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
        # uncentered_activation_center_position[0] = (mid_y - activation_shift_to_center[0]) % size_y
        # uncentered_activation_center_position[1] = (mid_x - activation_shift_to_center[1]) % size_x
        # activation_center_position_data[step] = uncentered_activation_center_position
        activation_center_position_data[step] = activation_shift_to_center * 1.0

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



def avg_speed_test(center_positions,SX,SY,t_1=150,t_2=2000,dt=10):
    
    
    
    p_x=center_positions[t_1:t_2,0]
    p_y=center_positions[t_1:t_2,1]
    
    #absolute pos

    dx=p_x[dt:]-p_x[:-dt]
    dy=p_y[dt:]-p_y[:-dt]
    

    dx_tot=dx
    dy_tot=dy
    


    speeds= np.sqrt(  (dx_tot)**2  +  (dy_tot)**2  )/dt
     

 
    avg_speed = speeds.sum()/(speeds.shape[0])
    
    return avg_speed


with open(os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/resources/creatures_categories.json", 'r') as f:
  data = json.load(f)



list_data=data[type_expe+"_exploration"]

lenia_config = Lenia_C.default_config()
lenia_config.SX = 256
lenia_config.SY = 256
lenia_config.final_step = 2000
lenia_config.version = 'pytorch_fft'
lenia_config.nb_kernels=10

initialization_space_config = Dict()
initialization_space = LeniaInitializationSpace(config=initialization_space_config)

system = Lenia_C(initialization_space=initialization_space, config=lenia_config, device='cuda')



pf=os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/resources/"+type_expe+"_exploration/"


total_seed=0

for key in list_data.keys():

    if(key[4]==str(crea_seed)):
        
        crea_info=list_data[key]
        if( (crea_info["is_robust"] and crea_info["is_soliton"] and crea_info["is_long_term_stable"])):
            total_seed+=1
print(total_seed)
            


performances=Dict()
ctr=0
print(sub_seed*(total_seed//10+1),(sub_seed+1)*(total_seed//10+1))
for key in list_data.keys():
    if(key[4]==str(crea_seed)):
        crea_info=list_data[key]


        if( (crea_info["is_robust"] and crea_info["is_soliton"] and crea_info["is_long_term_stable"])):
            
            
            if(ctr>=(sub_seed+1)*(total_seed//10+1)):
                break
            
            ctr+=1
            if(ctr-1>=sub_seed*(total_seed//10+1)):
                print(key)
                file=key+".pickle"
                stats=torch.load(pf+"stats/stats_"+file)

                perf=Dict()



                speed=avg_speed_test(stats['activation_center_position'],SX=256,SY=256)
                perf["speed_no_obs"]=speed

                #print(label.dtype)
                parameters=torch.load(pf+"parameters/"+file)
                policy_parameters=parameters["policy_parameters"]



                #static random


                abs_pos=stats["activation_center_position"]
                obs_pos=np.mod(abs_pos[1000],256)
                


                robu=0   
                list_statistics=[]
                #moving obstacles speed 1
                for i in range(50):
                    with torch.no_grad():
                        system.config.speed_x=-3
                        system.reset(initialization_parameters=policy_parameters['initialization'],
                                        update_rule_parameters=policy_parameters['update_rule'])


                        system.random_obstacle_bis(23,pos_obs=obs_pos)
                        system.generate_init_state()
                        # the method system.run() launches a rollout in Lenia
                        observations=system.run()
                        statistics=calc_statistics(observations["states"][:,:,:,0].cpu().numpy(),parameters)

                        crea_is_robust = bool(is_robust(statistics, low_mass_threshold=0, high_mass_threshold=6400, num_cells=256*256))

                        crea_is_long_term_stable= bool(is_long_term_stable(statistics,
                                                                             phase1_timepoints=(0, 1 / 4),
                                                                             phase2_timepoints=(3 / 4, 1),
                                                                             low_ratio_threshold=0.,
                                                                             high_ratio_threshold=2.0))
                        crea_is_soliton = bool(is_soliton(statistics, low_nr_objects=0, high_nr_objects=2, max_activity=0.6*256*256))

                        #is_moving = bool(is_moving(statistics, min_distance_from_init=100, final_step=1000))
                        if(crea_is_robust and crea_is_long_term_stable and crea_is_soliton):
                            robu+=1/50

                        list_statistics.append(statistics)
                torch.save(list_statistics,os.environ["SCRATCH"]+"/sensorimotor_lenia/resources/"+type_expe+"_exploration/tests/"+key+"_stats_tests_move1.pickle")
                perf["robustness_moving_obstacles3"]=robu


                
                
with open(os.environ["SCRATCH"]+"/sensorimotor_lenia/resources/"+type_expe+"_exploration/tests/"+'perfs_seed'+str(crea_seed)+'_subseed'+str(sub_seed)+'.json', 'w') as f:
    json.dump(performances, f)
 
print("finished")                
 
            



        






 



    
    
    
    
    











