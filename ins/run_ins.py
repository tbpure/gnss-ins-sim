import numpy as np
from matplotlib import pyplot as plt

from utils.data_io import get_imu_data_from_path, get_gps_data_from_path, get_att_data_from_path
from ins_algo import INS
file_path = '/Users/yangyu/PycharmProjects/gnss-ins-sim/cnn_gru/demo_saved_data/drone_sim/2025-06-03-09-45-29'

imu_data = get_imu_data_from_path(file_path)
gps_data = get_gps_data_from_path(file_path, ref = True)
att_data = get_att_data_from_path(file_path)
ins = INS(imu_data, gps_data, att_data[0][0:3])

len = len(imu_data)
for i in range(len):
    ins.step(imu_data)
result = ins.out_put
result_down_sample = np.array(result[::100])

plt.figure(figsize=(8, 8))
plt.plot(gps_data[:, 1], gps_data[:, 0], label='GPS', color='g')
plt.plot(result_down_sample[:, 1], result_down_sample[:, 0], label='INS', color='r')
plt.xlabel('Y (ECEF) [m]')
plt.ylabel('X (ECEF) [m]')
plt.title('2D Trajectory Comparison')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
