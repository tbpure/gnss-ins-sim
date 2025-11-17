import numpy as np

from sim_data_gen.gnss_ins_sim.attitude import attitude
from sim_data_gen.gnss_ins_sim.geoparams import geoparams


class INS(object):
    def __init__(self, imu_data, gps_data, init_vel, init_euler, dt = 0.01):
        # gps_data：[len, 3] 经纬度
        self.imu_data = imu_data
        # todo: 后续的数据需要在第一列时间戳，index需要从1开始
        self.dt = dt

        self.euler = init_euler
        self.c_bn = attitude.euler2dcm(self.euler)

        self.init_lla = gps_data[0][0:3]
        self.vel_b = self.c_bn.dot(init_vel)
        earth_param = geoparams.geo_param(self.init_lla)
        # self.position = geoparams.lla2ecef(self.init_lla)
        self.position = self.init_lla
        self.vel = init_vel
        self.g_n = np.array([0.0, 0.0, earth_param[2]])
        self.started = False
        self.freq_ratio = len(imu_data) / len(gps_data)
        self.out_put = []
        self.index = 0
        self.att = self.imu_data[self.index][0:3]

    def step(self, imu_data):
        if self.index == 0:
            self.out_put.append(self.position)
        else:
            c_bn = self.c_bn
            acc = imu_data[self.index - 1][0:3]
            gyro = imu_data[self.index - 1][3:6]
            self.euler = attitude.euler_update_zyx(self.euler, gyro, self.dt)
            self.vel_b = self.vel_b + (acc + c_bn.dot(self.g_n)) * self.dt -\
                attitude.cross3(gyro, self.vel_b) * self.dt # 科氏项
            self.c_bn = attitude.euler2dcm(self.euler)
            self.vel = self.c_bn.T.dot(self.vel_b)

            # 使用上一时刻的速度更新位置
            self.position = self.position + self.vel * self.dt
            self.out_put.append(self.position)
        self.index += 1


if __name__ == "__main__":
    pass

