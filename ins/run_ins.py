import numpy as np
from matplotlib import pyplot as plt

from sim_data_gen.gnss_ins_sim.geoparams import geoparams
from sim_data_gen.gnss_ins_sim.geoparams.geoparams import geo_param
from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path, get_att_data_from_path, get_ref_data_from_path
from ins_algo import INS
from utils.unit_transfer import deg2rad


file_path = "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/default/2025-11-24-16-26-53"
imu_data = get_imu_data_from_path(file_path, ref=False)
imu_data[:, 3:6] = deg2rad(imu_data[:, 3:6])
gps_data = get_gnss_data_from_path(file_path, ref=True)
att_data = get_att_data_from_path(file_path, ref=True)
ref_vel = get_ref_data_from_path(file_path, ['vel'])
if gps_data.shape[1] != 3:
    init_vel_b = gps_data[:, :3]
else:
    init_vel_b = ref_vel['vel'][0, 0:3]
ins = INS(imu_data, gps_data, init_vel_b, deg2rad(att_data[0][0:3]))

len = len(imu_data)
for i in range(len):
    ins.step()
result = ins.out_put
result_down_sample = np.array(result[::100])

# gps_ecef = np.array([geoparams.lla2ecef(llh[:3]) for llh in gps_data])
# result = np.array([geoparams.lla2ecef(flip) for flip in result_down_sample])
plt.figure(figsize=(8, 8))
# plt.plot(gps_ecef[:, 1], gps_ecef[:, 0], label='GPS (ECEF)', color='g')
# for i in range(gps_data.shape[0]):
#     gps_data[i][0:3] = geoparams.lla2ecef(gps_data[i][0:3])

plt.plot(gps_data[:, 1], gps_data[:, 0], 'b-', label='GPS (ECEF)')
plt.plot(np.array(result)[:, 1], np.array(result)[:, 0], 'r--', label='INS (ECEF)')
plt.xlabel('Y (ECEF) [m]')
plt.ylabel('X (ECEF) [m]')
plt.title('2D Trajectory Comparison')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
