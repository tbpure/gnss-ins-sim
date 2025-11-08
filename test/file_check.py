from sim_data_gen.gnss_ins_sim.attitude import attitude
from utils.data_io import get_imu_data_from_path, get_att_data_from_path


def check_acc():
    path = '/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2025-11-08-13-39-04'

    ref_imu = get_imu_data_from_path(path, ref=True)
    vel_imu = get_imu_data_from_path(path, ref=False)

    att_ref = get_att_data_from_path(path, ref=True)
    for index in range(len(ref_imu)):
        cbn = attitude.euler2dcm(att_ref[index][0:3])
        vel_b = vel_imu[index][0:3]
        acc = cbn.T.dot(vel_b)
        ref_acc = ref_imu[index][0:3]

if __name__ == '__main__':
    check_acc()