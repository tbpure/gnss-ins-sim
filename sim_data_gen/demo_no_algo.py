# -*- coding: utf-8 -*-
# Filename: demo_no_algo.py

"""
The simplest demo of Sim.
Only generate reference trajectory (pos, vel, sensor output). No algorithm.
Created on 2018-01-23
@author: dongxiaoguang
"""

import os
import math

import numpy as np

from sim_data_gen.gnss_ins_sim.sim import imu_model
from sim_data_gen.gnss_ins_sim.sim import ins_sim


def test_path_gen(save_to_dir:str='', target_file_path:str='', file_name:str='motion_def-3d.csv'):
    # globals
    D2R = math.pi / 180

    if target_file_path == '':
        motion_def_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sim_files/def_motion/'))
    else:
        motion_def_path = target_file_path
    save_to_dir = save_to_dir + '/' + file_name.removesuffix('.csv')
    fs = 100.0  # IMU sample frequency
    fs_gps = 1.0  # GPS sample frequency
    fs_mag = fs  # magnetometer sample frequency, not used for now

    '''
    test only path generation in Sim.
    '''
    #### choose a built-in IMU model, typical for IMU381
    imu_err = 'mid-accuracy'
    # imu_err = {
    #     'gyro_b': np.array([0.0, 0.0, 0.0]),
    #     'gyro_arw': np.array([0.8, 0.8, 0.8]),  # 陀螺角随机游走
    #     'gyro_b_stability': np.array([15.0, 15.0, 15.0]),  # 零偏稳定性（°/h）
    #     'gyro_b_corr': np.array([200.0, 200.0, 200.0]),
    #
    #     'accel_b': np.array([0.0, 0.0, 0.0]),
    #     'accel_vrw': np.array([0.08, 0.08, 0.08]),  # 加速度计速度随机游走
    #     'accel_b_stability': np.array([5e-4, 5e-4, 5e-4]),  # 零偏漂移
    #     'accel_b_corr': np.array([200.0, 200.0, 200.0]),
    #
    #     'mag_std': np.array([0.3, 0.3, 0.3])
    # }
    # generate GPS and magnetometer data
    imu = imu_model.IMU(axis=9, gps=True)

    #### start simulation
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      motion_def_path + "//" + file_name,
                      ref_frame=1,
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)
    # save simulation data to files
    sim.results(save_to_dir)
    # plot data, 3d plot of reference positoin, 2d plots of gyro and accel
    # sim.plot(['ref_pos', 'gyro', 'gps_visibility'], opt={'ref_pos': '3d'})

if __name__ == '__main__':
    test_path_gen('./sim_files/saved_file/', '', 'default.csv')
