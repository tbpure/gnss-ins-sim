# -*- coding: utf-8 -*-
# Filename: demo_free_integration.py

"""
A simple free integration (strapdown inertial navigation) demo of Sim.
Created on 2018-01-23
@author: dongxiaoguang
"""

import os
import math
import numpy as np
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180

motion_def_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sim_files/def_motion/'))
fs = 100.0          # IMU sample frequency

def test_free_integration(save_to_path:str, file_name: str = 'default.csv'):
    '''
    test Sim
    '''
    #### IMU model, typical for IMU381
    save_to_path = save_to_path + '/' + file_name.removesuffix('.csv')
    imu_err = {
        'gyro_b': np.array([0.0, 0.0, 0.0]),
        'gyro_arw': np.array([0.8, 0.8, 0.8]),  # 陀螺角随机游走
        'gyro_b_stability': np.array([15.0, 15.0, 15.0]),  # 零偏稳定性（°/h）
        'gyro_b_corr': np.array([200.0, 200.0, 200.0]),

        'accel_b': np.array([0.0, 0.0, 0.0]),
        'accel_vrw': np.array([0.08, 0.08, 0.08]),  # 加速度计速度随机游走
        'accel_b_stability': np.array([5e-4, 5e-4, 5e-4]),  # 零偏漂移
        'accel_b_corr': np.array([200.0, 200.0, 200.0]),

        'mag_std': np.array([0.3, 0.3, 0.3])
    }
    odo_err = {'scale': 0.999,
               'stdv': 0.1}
    # do not generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err, axis=9, gps=False, odo=True, odo_opt=odo_err)

    #### Algorithm
    # Free integration in a virtual inertial frame
    from demo_algorithms import free_integration_odo
    from demo_algorithms import free_integration
    '''
    Free integration requires initial states (position, velocity and attitude). You should provide
    theses values when you create the algorithm object.
    '''
    ini_pos_vel_att = np.genfromtxt(motion_def_path+"//" + file_name,\
                                    delimiter=',', skip_header=1, max_rows=1)
    ini_pos_vel_att[0] = ini_pos_vel_att[0] * D2R
    ini_pos_vel_att[1] = ini_pos_vel_att[1] * D2R
    ini_pos_vel_att[6:9] = ini_pos_vel_att[6:9] * D2R
    # add initial states error if needed
    ini_vel_err = np.array([0.0, 0.0, 0.0]) # initial velocity error in the body frame, m/s
    ini_att_err = np.array([0.0, 0.0, 0.0]) # initial Euler angles error, deg
    ini_pos_vel_att[3:6] += ini_vel_err
    ini_pos_vel_att[6:9] += ini_att_err * D2R
    # create the algorith object
    algo1 = free_integration_odo.FreeIntegration(ini_pos_vel_att)
    algo2 = free_integration.FreeIntegration(ini_pos_vel_att)

    #### start simulation
    sim = ins_sim.Sim([fs, 0.0, 0.0],
                      motion_def_path+"//" + file_name,
                      ref_frame=1,
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=[algo2])
    # run the simulation for 1000 times
    sim.run(1)
    # generate simulation results, summary
    # do not save data since the simulation runs for 1000 times and generates too many results
    sim.results(data_dir=save_to_path, err_stats_start=-1, gen_kml=False)
    # plot postion error
    # sim.plot(['pos'], opt={'pos':'error'})
    sim.plot(['ref_pos'], opt={'ref_pos': '3d'})
    sim.plot(['pos'], opt={'pos': '3d'})

if __name__ == '__main__':
    test_free_integration(save_to_path='./sim_files/saved_file/', file_name='default.csv')
