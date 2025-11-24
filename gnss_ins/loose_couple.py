import matplotlib.pyplot as plt

from KF.ekf_base import EKF
from ins.ins_algo import INS
import numpy as np

from sim_data_gen.gnss_ins_sim.attitude import attitude
from sim_data_gen.gnss_ins_sim.geoparams import geoparams
from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path, get_att_data_from_path, \
    get_ref_data_from_path
from utils.matrix_utils import skew
from utils.unit_transfer import deg2rad

STATUS_DIMENSION = 21

IDX_DR = slice(0, 3) # 位置误差
IDX_DV = slice(3, 6) # 速度误差
IDX_PHI = slice(6, 9) # 姿态误差
IDX_BG = slice(9, 12) # 陀螺零偏误差
IDX_BA = slice(12, 15) # 加速度计零偏误差
IDX_SG = slice(15, 18) # 陀螺比例因子误差
IDX_SA = slice(18, 21) # 加速度计比例因子误差


# WGS-84 常量
WGS84_A = 6378137.0               # 长半轴 a (m)
WGS84_E2 = 6.69437999014e-3       # 偏心率平方 e^2
OMEGA_IE = 7.292115e-5            # 地球自转角速度 (rad/s)

def calc_RN_RM(lat_rad):
    s2 = np.sin(lat_rad)**2
    RN = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s2)
    RM = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2)**1.5
    return RN, RM

def calc_omega_ie_n(lat_rad):
    # 在 NED（导航系）下的地球自转向量分量（rad/s）
    # omega_ie^n = [omega_ie * cos(lat), 0, -omega_ie * sin(lat)]
    return OMEGA_IE * np.array([np.cos(lat_rad), 0.0, -np.sin(lat_rad)])

def calc_omega_en_n(vn, lat_rad, RN, RM, h):
    # vn = [vN, vE, vD]  (m/s)
    vN, vE, vD = vn
    denom_RN = RN + h
    denom_RM = RM + h
    # 注意符号约定 (NED)：下面表达式是常用教材中的形式
    omega_en_n = np.array([
        vE / denom_RN,
        -vN / denom_RM,
        - (vE * np.tan(lat_rad)) / denom_RN
    ])
    return omega_en_n

def calc_gravity(lat_rad, h):
    # Somigliana 公式的近似（m/s^2）
    g_e = 9.7803253359
    k = 0.00193185265241
    s2 = np.sin(lat_rad)**2
    gamma = g_e * (1.0 + k * s2) / np.sqrt(1.0 - WGS84_E2 * s2)
    # 高度线性修正项（近似）
    gamma_h = gamma - 3.086e-6 * h
    # 返回 NED 方向的重力向量（向下为正或负取决于约定，常见取 g_n = [0,0,-g]）
    return np.array([0.0, 0.0, -gamma_h])

def build_earth(lat_rad, h, vn):
    RN, RM = calc_RN_RM(lat_rad)
    omega_ie_n = calc_omega_ie_n(lat_rad)
    omega_en_n = calc_omega_en_n(vn, lat_rad, RN, RM, h)
    omega_in_n = omega_ie_n + omega_en_n
    g_n = calc_gravity(lat_rad, h)
    earth = {
        "lat": lat_rad,
        "h": h,
        "RN": RN,
        "RM": RM,
        "omega_ie_n": omega_ie_n,
        "omega_en_n": omega_en_n,
        "omega_in_n": omega_in_n,
        "g_n": g_n
    }
    return earth


class LooseCouple:
    def __init__(self, imu_params, imu_data, gnss_data, init_vel, init_euler):
        self.dt = 0.01
        self.imu_params = imu_params
        self.imu_data = imu_data
        self.gnss_data = gnss_data
        self.ins = INS(imu_data, gnss_data, init_vel, init_euler)
        ekf = EKF(STATUS_DIMENSION)


        # todo
        # 初始化状态协方差矩阵
        P = np.eye(STATUS_DIMENSION)
        # 惯导位置误差
        P[IDX_DR, IDX_DR] *= 1.0 ** 2
        # 惯导速度误差
        P[IDX_DV, IDX_DV] *= 0.1 ** 2
        # 姿态误差
        P[IDX_PHI, IDX_PHI] *= np.deg2rad(0.1) ** 2
        P[IDX_BG, IDX_BG] *= np.deg2rad(0.1 / 3600) ** 2
        P[IDX_BA, IDX_BA] *= (1e-3 * imu_params["G_CONST"]) ** 2
        ekf.set_initial_state(np.zeros((STATUS_DIMENSION, 1)), P)

        self.c_bn = attitude.euler2dcm(init_euler)
        self.init_lla = gps_data[0][0:3]
        self.vel_b = self.c_bn.dot(init_vel)
        earth_param = geoparams.geo_param(geoparams.ecef2lla(self.init_lla))
        self.g_n = np.array([0.0, 0.0, earth_param[2]])
        # 零偏
        self.gyro_bias = imu_params["gyro_bias"]
        self.accel_bias = imu_params["accel_bias"]

        # 标准差
        self.gyro_std = imu_params["gyro_std"]
        self.accel_std = imu_params["accel_std"]

        # 比例因子 int
        self.gyro_scale = imu_params["gyro_scale"]
        self.accel_scale = imu_params["accel_scale"]


    def build_F(self, Cnb, acc, gyro, vn, ecef_pos, earth, imu_params):
        """
        # pos：lla坐标系
        构造完整 21x21 F 矩阵
        """

        F = np.zeros((STATUS_DIMENSION, STATUS_DIMENSION))
        lla_pos = geoparams.ecef2lla(ecef_pos)
        # 地球/位置参数
        RM, RN = earth["RM"], earth["RN"]
        h, phi = lla_pos[2], lla_pos[0]  # phi表示纬度
        vN, vE, vD = vn

        # 简化：假设 pos[0] (纬度) 和 omega_in_n 来自 earth 字典
        phi_lat = earth.get("lat", 0.0)  # 真实的纬度，其实与上面的phi是相通的？
        omega_in_n = earth.get("omega_in_n", np.zeros(3))

        # ----------------------------------------
        # F_rr 位置误差微分
        # ----------------------------------------
        Frr = np.array([
            [-vD / (RM + h), 0, vN / (RM + h)],
            [vE * np.tan(phi_lat) / (RN + h), -(vD - vN * np.tan(phi_lat)) / (RN + h), vE / (RN + h)],
            [0, 0, 0]
        ])
        # 简化 F_rr, 假设东北天坐标系下位置误差的微分
        # Frr = np.zeros((3, 3))  # 在简单仿真中，此项影响小，暂设为0

        F[IDX_DR, IDX_DR] = Frr
        F[IDX_DR, IDX_DV] = np.eye(3)

        # ----------------------------------------
        # F_vr, F_vv (速度误差)
        # ----------------------------------------
        # F_vr (重力异常和牵连加速度项)
        Fvr = np.zeros((3, 3))
        # 简化重力模型 (垂向)
        Fvr[2, 2] = 2 * self.g_n[2] * -1 / (RM + h)
        F[IDX_DV, IDX_DR] = Fvr

        # F_vv (科里奥利力项)
        Fvv = -skew(2 * omega_in_n)
        F[IDX_DV, IDX_DV] = Fvv

        # 姿态误差耦合加速度项（关键项）
        F[IDX_DV, IDX_PHI] = -Cnb @ skew(acc)

        # 加速度计零偏与比例因子耦合
        F[IDX_DV, IDX_BA] = Cnb
        F[IDX_DV, IDX_SA] = Cnb @ np.diag(acc)

        # ----------------------------------------
        # 姿态误差方程：ϕ̇ = -ω_in× ϕ + δω_in
        # ----------------------------------------
        F[IDX_PHI, IDX_PHI] = -skew(omega_in_n)

        # 陀螺零偏与比例因子耦合
        F[IDX_PHI, IDX_BG] = -Cnb
        F[IDX_PHI, IDX_SG] = -Cnb @ np.diag(gyro)

        # ----------------------------------------
        # 传感器一阶马尔科夫过程
        F[IDX_BG, IDX_BG] = -(1.0 / imu_params["Tgb"]) * np.eye(3)
        F[IDX_BA, IDX_BA] = -(1.0 / imu_params["Tab"]) * np.eye(3)
        F[IDX_SG, IDX_SG] = -(1.0 / imu_params["Tgs"]) * np.eye(3)
        F[IDX_SA, IDX_SA] = -(1.0 / imu_params["Tas"]) * np.eye(3)

        return F


    def build_G(self, Cnb):
        G = np.zeros((STATUS_DIMENSION, 18))
        G[IDX_DV, 0:3] = Cnb
        G[IDX_PHI, 3:6] = Cnb
        G[IDX_BG, 6:9] = np.eye(3)
        G[IDX_BA, 9:12] = np.eye(3)
        G[IDX_SG, 12:15] = np.eye(3)
        G[IDX_SA, 15:18] = np.eye(3)
        return G


    def build_q(self, imu_params):
        q = np.zeros((18, 18))
        q[0:3, 0:3] = (imu_params["ARW"] ** 2) * np.eye(3)
        q[3:6, 3:6] = (imu_params["VRW"] ** 2) * np.eye(3)
        q[6:9, 6:9] = 2.0 * (imu_params["sigma_bg"] ** 2) / imu_params["Tgb"] * np.eye(3)
        q[9:12, 9:12] = 2.0 * (imu_params["sigma_ba"] ** 2) / imu_params["Tab"] * np.eye(3)
        q[12:15, 12:15] = 2.0 * (imu_params["sigma_sg"] ** 2) / imu_params["Tgs"] * np.eye(3)
        q[15:18, 15:18] = 2.0 * (imu_params["sigma_sa"] ** 2) / imu_params["Tas"] * np.eye(3)
        return q


    def run(self):
        # 初始化
        saved = self.saved
        ins = self.ins
        imu_params = self.imu_params.copy()
        gnss_data = self.gnss_data.copy()
        ins_data = self.imu_data.copy()

        x = np.zeros((STATUS_DIMENSION, 1))
        P = np.eye(STATUS_DIMENSION)
        P[IDX_DR, IDX_DR] *= 1.0 ** 2
        P[IDX_DV, IDX_DV] *= 0.1 ** 2
        P[IDX_PHI, IDX_PHI] *= np.deg2rad(0.1) ** 2
        P[IDX_BG, IDX_BG] *= np.deg2rad(0.1 / 3600) ** 2

        P[IDX_BA, IDX_BA] *= (1e-3 * self.g_n[2] * -1) ** 2
        imu_params["ARW"] = self.imu_params["ARW"] / np.sqrt(1 / 0.01)  # 假设 ARW/VRW 单位是 /sqrt(s)
        imu_params["VRW"] = self.imu_params["VRW"] / np.sqrt(1 / 0.01)

        # 修正：假设 ARW/VRW 单位已经是 (m/s^2)/sqrt(Hz) 或 (rad/s)/sqrt(Hz)
        # 那么 Q = sigma^2 / dt。 而连续时间 q = sigma^2 * dt。
        # 讲义中 ARW 是 sqrt(q_a * dt)，所以 q_a = ARW^2 / dt
        # 不，ARW/VRW 本身就是 sqrt(PSD)
        # q[0:3, 0:3] = (imu_params["ARW"] ** 2) * eye(3) # PSD = (m/s^2/sqrt(Hz))^2 = (m/s^2)^2/Hz
        q = self.build_q(imu_params)

        R_pos = np.eye(3) * 1.5 ** 2  # GNSS position variance (m^2)
        R_vel = np.eye(3) * 0.5 ** 2  # GNSS velocity variance (m^2/s^2)

        results = []
        dt = self.dt
        for i, imu in enumerate(ins_data):
            # ins状态更新
            self.ins.step()
            # EKF 使用的输入：来自漂移的 INS 和带噪声的 IMU 读数

            acc = ins.acc
            gyro = ins.gyro
            Cnb = ins.c_bn  # 漂移的姿态
            vn = ins.vel  # 漂移的速度
            pos = ins.position  # 漂移的位置

            # =====================================
            # ① 预测：IMU 100Hz 更新
            # =====================================
            lla = geoparams.ecef2lla(ins.position)
            earth = build_earth(lla[0], lla[2], ins.vel)
            F = self.build_F(Cnb, acc, gyro, vn, pos, earth, imu_params)
            G = self.build_G(Cnb)

            if np.any(np.isnan(F)) or np.any(np.isinf(F)):
                print("F ERROR", F)
                raise ValueError("F invalid")

            Phi = np.eye(STATUS_DIMENSION) + F * dt

            # 离散化 Qk (近似)：Qk = Phi * G * q * G.T * Phi.T * dt
            # 修正：Qk = G * q * G.T * dt (一阶近似)
            if np.any(np.isnan(G)) or np.any(np.isinf(G)):
                print("G ERROR", G)
            if np.any(np.isnan(q)) or np.any(np.isinf(q)):
                print("q ERROR", q)

            if np.linalg.norm(G) > 1e6:
                print("G too large", G)
            if np.linalg.norm(q) > 1e6:
                print("q too large", q)
            Qk = (G @ q @ G.T) * dt

            # 状态更新
            x = Phi @ x
            if np.any(np.isnan(Phi)) or np.any(np.isinf(Phi)):
                print("Phi ERROR", Phi)
            P = Phi @ P @ Phi.T + Qk

            # =====================================
            # ② 若此时有 GNSS 数据则进行更新
            # =====================================
            if i % 100 == 0:
                gnss_index = int (i / 100)
                gnss = gnss_data[gnss_index][0:3]
                # -----------------------
                # GNSS 位置观测
                # -----------------------
                lb = np.zeros(3)
                H_pos = np.zeros((3, STATUS_DIMENSION))
                H_pos[:, IDX_DR] = np.eye(3)
                H_pos[:, IDX_PHI] = skew(Cnb @ lb)

                # 观测 z = INS 预测 - GNSS 测量
                z_pos = (ins.position - gnss).reshape(3, 1)

                # 创新 y = z - H * x_pred
                y = z_pos - H_pos @ x
                S = H_pos @ P @ H_pos.T + R_pos
                K = P @ H_pos.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(STATUS_DIMENSION) - K @ H_pos) @ P

                # -----------------------
                # GNSS 速度观测
                # -----------------------
                H_vel = np.zeros((3, STATUS_DIMENSION))
                H_vel[:, IDX_DV] = np.eye(3)
                term = Cnb @ np.cross(lb, gyro)
                H_vel[:, IDX_PHI] = -skew(term)

                z_vel = ins.vel - gnss_data[gnss_index][3:6]
                y = z_vel - H_vel @ x
                S = H_vel @ P @ H_vel.T + R_vel
                K = P @ H_vel.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(STATUS_DIMENSION) - K @ H_vel) @ P


                # --- 状态反馈 (可选，若 INS 循环在外部) ---
                # 此处 EKF 是外部运行的，我们只保存误差状态 x
                # 在一个真实系统中，x 会被反馈回 INS 循环以修正 ins_pos, ins_vel

            # 保存结果（每个 IMU 时刻都保存）
            results.append(x.copy())

        return results

def test():

    pass


if __name__ == "__main__":
    file_path = "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2025-11-10-20-00-21"
    imu_params = {
        "G_CONST": 9.8,
        "ARW": 0.1,
        "VRW": 0.1,
        "gyro_scale": 1.0,
        "accel_scale": 1.0,
        "gyro_std": 0.1,
        "accel_std": 0.1,
        "gyro_bias": 0.1,
        "accel_bias": 0.1,
        "sigma_bg": 0.1,
        "sigma_ba": 0.1,
        "sigma_sg": 0.1,
        "sigma_sa": 0.1,
        "Tgb": 0.01,
        "Tab": 0.01,
        "Tgs": 0.01,
        "Tas": 0.01,
    }
    gnss_params = {
        "pos_std": 1.0,
        "vel_std": 1.0,
    }
    imu_data = get_imu_data_from_path(file_path, ref=False)
    imu_data[:, 3:6] = np.deg2rad(imu_data[:, 3:6])
    gps_data = get_gnss_data_from_path(file_path, ref=False)
    att_data = get_att_data_from_path(file_path, ref=True)
    ref_vel = get_ref_data_from_path(file_path, ['vel'])
    ref_pos = get_gnss_data_from_path(file_path, ref = True)
    if gps_data.shape[1] != 3:
        init_vel_b = gps_data[0, 3:6]
    else:
        init_vel_b = ref_vel['vel'][0, 0:3]
    loose = LooseCouple(imu_params, imu_data, gps_data, init_vel_b, deg2rad(att_data[0][0:3]))

    ekf_states = loose.run()
    ins_result = loose.ins.out_put
    ekf_pos = np.zeros_like(ins_result)
    for i in range(len(ekf_states)):
        s = ekf_states[i]

        # 注意：s[IDX_DR, 0] 是 EKF 对 δr (位置误差) 的估计
        # P_corrected = P_ins - δr_estimated
        ekf_pos[i] = ins_result[i] - s[IDX_DR, 0]
    ins_result = np.array(ins_result)
    plt.figure(figsize=(8, 8), dpi=120)

    plt.plot(ekf_pos[:, 1], ekf_pos[:, 0], label="EKF", linewidth=2)
    plt.plot(ins_result[:, 1], ins_result[:, 0], label="INS", linewidth=2, alpha=0.8)
    plt.plot(gps_data[:, 1], gps_data[:, 0], label="GPS", linestyle='--', linewidth=1.8, alpha=0.9)
    plt.plot(ref_pos[:, 1], ref_pos[:, 0], label="Reference", linewidth=1.5)

    plt.xlabel('Y (ECEF) [m]', fontsize=12)
    plt.ylabel('X (ECEF) [m]', fontsize=12)

    plt.title('2D Trajectory Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=11)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()