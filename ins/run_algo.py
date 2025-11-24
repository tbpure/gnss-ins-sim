import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sim_data_gen.demo_algorithms import free_integration
from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path, get_data_from_path
from utils.unit_transfer import deg2rad

# globals
D2R = math.pi/180

ini_pos_vel_att = np.genfromtxt('/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/def_motion/motion_def-90deg_turn_long.csv', \
                                delimiter=',', skip_header=1, max_rows=1)
ini_pos_vel_att[0] = ini_pos_vel_att[0] * D2R
ini_pos_vel_att[1] = ini_pos_vel_att[1] * D2R
ini_pos_vel_att[6:9] = ini_pos_vel_att[6:9] * D2R
# add initial states error if needed
ini_vel_err = np.array([0.0, 0.0, 0.0])  # initial velocity error in the body frame, m/s
ini_att_err = np.array([0.0, 0.0, 0.0])  # initial Euler angles error, deg
ini_pos_vel_att[3:6] += ini_vel_err
ini_pos_vel_att[6:9] += ini_att_err * D2R

file_path = '/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2025-11-08-19-10-43'
# create the algorith object
algo = free_integration.FreeIntegration(ini_pos_vel_att)
imu_data = get_imu_data_from_path(file_path, ref=False)
gps = get_gnss_data_from_path(file_path, ref=True)
acc = imu_data[:, 0:3]
gyro_degree = imu_data[:, 3:6]
gyro = deg2rad(gyro_degree)
var = pd.read_csv(file_path + '/pos-algo0_0.csv').to_numpy()

# algo_input = [1, 100, acc, gyro]
algo_input = [1, 100, gyro, acc]
algo.run(algo_input)
ins_pos = algo.pos.copy()
# 假设位置格式都是 [x, y, z] 或 [E, N, U]
plt.figure(figsize=(8, 6))
plt.plot(gps[:, 1], gps[:, 0], 'r-', label='GPS reference')
plt.plot(ins_pos[:, 1], ins_pos[:, 0], 'b--', label='INS integration')
plt.plot(var[:, 1], var[:, 0], 'b--', label='INS integration')

plt.title('INS vs GPS Trajectory')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()