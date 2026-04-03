import numpy as np
from matplotlib import pyplot as plt

from ins.run_algo import D2R
from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path, get_att_data_from_path, \
    get_ref_data_from_path
# 假设这些从你的前文导入
from utils.matrix_utils import skew
from sim_data_gen.gnss_ins_sim.geoparams import geoparams
from gnss_ins.loose_couple import calc_RN_RM, calc_omega_en_n, calc_omega_ie_n, calc_gravity
from utils.unit_transfer import deg2rad

STATUS_DIMENSION = 21
IDX_DR = slice(0, 3)
IDX_DV = slice(3, 6)
IDX_PHI = slice(6, 9)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)
IDX_SG = slice(15, 18)
IDX_SA = slice(18, 21)


class StrongTrackingLooseCouple:
    """
    基于强跟踪扩展卡尔曼滤波 (STEKF) 的松组合导航。
    采用单渐消因子 (Single Fading Factor) 调整预测协方差 P，以提升突变状态下的跟踪能力。
    """

    def __init__(self, imu_params, imu_data, gnss_data, init_vel, init_euler,
                 stf_alpha=0.95, dt=0.01, save=None):
        """
        stf_alpha: 新息协方差的指数平滑因子（通常取 0.9 ~ 0.99）
        """
        if save is None:
            self.save = []
        else:
            self.save = save

        self.imu_params = imu_params.copy()
        self.imu_data = imu_data.copy()
        self.gnss_data = gnss_data.copy()
        self.dt = dt

        from ins.ins_algo import INS
        self.ins = INS(imu_data, gnss_data, init_vel, init_euler)

        self.x = np.zeros((STATUS_DIMENSION, 1))
        self.P = np.eye(STATUS_DIMENSION)
        self.P[IDX_DR, IDX_DR] *= 1.0 ** 2
        self.P[IDX_DV, IDX_DV] *= 0.1 ** 2
        self.P[IDX_PHI, IDX_PHI] *= np.deg2rad(0.1) ** 2
        self.P[IDX_BG, IDX_BG] *= np.deg2rad(0.1 / 3600) ** 2
        self.P[IDX_BA, IDX_BA] *= (1e-3 * imu_params["G_CONST"]) ** 2

        # 固定的观测噪声 (位置和速度)
        self.R_pos = np.eye(3) * 1.5 ** 2
        self.R_vel = np.eye(3) * 0.5 ** 2

        # 为了 STEKF 方便，将位置和速度合并为 6 维观测噪声矩阵
        self.R_gnss = np.zeros((6, 6))
        self.R_gnss[0:3, 0:3] = self.R_pos
        self.R_gnss[3:6, 3:6] = self.R_vel

        # STEKF 参数
        self.stf_alpha = stf_alpha
        self.Vk = self.R_gnss.copy()  # 初始化实际新息协方差为 R

        self.saved = {}
        self.results = []

    def build_F(self, Cnb, acc, gyro, vn, ecef_pos, earth, imu_params):
        # 保持与原版一致
        F = np.zeros((STATUS_DIMENSION, STATUS_DIMENSION))
        lla_pos = geoparams.ecef2lla(ecef_pos)
        RM, RN = earth["RM"], earth["RN"]
        h, phi = lla_pos[2], lla_pos[0]
        vN, vE, vD = vn
        phi_lat = earth.get("lat", 0.0)
        omega_in_n = earth.get("omega_in_n", np.zeros(3))

        Frr = np.array([
            [-vD / (RM + h), 0, vN / (RM + h)],
            [vE * np.tan(phi_lat) / (RN + h), -(vD - vN * np.tan(phi_lat)) / (RN + h), vE / (RN + h)],
            [0, 0, 0]
        ])
        F[IDX_DR, IDX_DR] = Frr
        F[IDX_DR, IDX_DV] = np.eye(3)

        Fvr = np.zeros((3, 3))
        Fvr[2, 2] = 2 * (self.ins.g_n[2]) * -1 / (RM + h)
        F[IDX_DV, IDX_DR] = Fvr

        Fvv = -skew(2 * omega_in_n)
        F[IDX_DV, IDX_DV] = Fvv

        F[IDX_DV, IDX_PHI] = -Cnb @ skew(acc)
        F[IDX_DV, IDX_BA] = Cnb
        F[IDX_DV, IDX_SA] = Cnb @ np.diag(acc)

        F[IDX_PHI, IDX_PHI] = -skew(omega_in_n)
        F[IDX_PHI, IDX_BG] = -Cnb
        F[IDX_PHI, IDX_SG] = -Cnb @ np.diag(gyro)

        F[IDX_BG, IDX_BG] = -(1.0 / self.imu_params["Tgb"]) * np.eye(3)
        F[IDX_BA, IDX_BA] = -(1.0 / self.imu_params["Tab"]) * np.eye(3)
        F[IDX_SG, IDX_SG] = -(1.0 / self.imu_params["Tgs"]) * np.eye(3)
        F[IDX_SA, IDX_SA] = -(1.0 / self.imu_params["Tas"]) * np.eye(3)

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
        ins = self.ins
        imu_params = self.imu_params.copy()
        gnss_data = self.gnss_data.copy()
        ins_data = self.imu_data.copy()

        x = self.x
        P = self.P
        q = self.build_q(imu_params)
        dt = self.dt
        saved = {}

        for i, imu in enumerate(ins_data):
            # 1) INS propagate
            ins.step()
            acc = ins.acc
            gyro = ins.gyro
            Cnb = ins.c_bn
            vn = ins.vel
            pos = ins.position

            # 2) 构造 F, G, Qk
            lla = geoparams.ecef2lla(pos)
            earth = {
                "lat": lla[0],
                "h": lla[2],
                "RN": calc_RN_RM(lla[0])[0],
                "RM": calc_RN_RM(lla[0])[1],
                "omega_en_n": calc_omega_en_n(vn, lla[0], *calc_RN_RM(lla[0]), lla[2]),
            }
            earth["omega_ie_n"] = calc_omega_ie_n(lla[0])
            earth["omega_in_n"] = earth["omega_ie_n"] + earth["omega_en_n"]
            earth["g_n"] = calc_gravity(lla[0], lla[2])

            F = self.build_F(Cnb, acc, gyro, vn, pos, earth, imu_params)
            G = self.build_G(Cnb)

            Phi = np.eye(STATUS_DIMENSION) + F * dt
            Qk = (G @ q @ G.T) * dt

            # 3) Predict
            x = Phi @ x
            P = Phi @ P @ Phi.T + Qk

            # 4) GNSS Update
            if i % 100 == 0:
                gnss_index = int(i / 100)
                gnss_pos = gnss_data[gnss_index][0:3]
                gnss_vel = gnss_data[gnss_index][3:6]

                lb = np.zeros(3)

                # --- 构造 6 维联合观测矩阵 H_gnss ---
                H_pos = np.zeros((3, STATUS_DIMENSION))
                H_pos[:, IDX_DR] = np.eye(3)
                H_pos[:, IDX_PHI] = skew(Cnb @ lb)
                z_pos = (ins.position - gnss_pos).reshape(3, 1)

                H_vel = np.zeros((3, STATUS_DIMENSION))
                H_vel[:, IDX_DV] = np.eye(3)
                term = Cnb @ np.cross(lb, gyro)
                H_vel[:, IDX_PHI] = -skew(term)
                z_vel = (ins.vel - gnss_vel).reshape(3, 1)

                H_gnss = np.vstack((H_pos, H_vel))
                z_gnss = np.vstack((z_pos, z_vel))

                # 当前新息
                y_gnss = z_gnss - H_gnss @ x

                # --- STEKF: 计算并引入渐消因子 ---
                # 1. 平滑更新实际新息协方差 Vk
                self.Vk = self.stf_alpha * self.Vk + (1 - self.stf_alpha) * (y_gnss @ y_gnss.T)

                # 2. 计算渐消因子 lambda
                # tr_N = tr(Vk - R)
                # tr_M = tr(H * P * H^T)
                tr_N = np.trace(self.Vk - self.R_gnss)
                tr_M = np.trace(H_gnss @ P @ H_gnss.T)

                if tr_M > 0:
                    lambda_k = max(1.0, tr_N / tr_M)
                else:
                    lambda_k = 1.0

                # 3. 强跟踪：放大预测协方差
                P = lambda_k * P

                if 'lambda' in self.save:
                    saved.setdefault('lambda', []).append(lambda_k)

                # --- 常规 EKF 状态更新 ---
                S_gnss = H_gnss @ P @ H_gnss.T + self.R_gnss
                K_gnss = P @ H_gnss.T @ np.linalg.inv(S_gnss)

                x = x + K_gnss @ y_gnss
                P = (np.eye(STATUS_DIMENSION) - K_gnss @ H_gnss) @ P

                if 'P' in self.save:
                    saved.setdefault('P', []).append(P.copy())

            self.results.append(x.copy())

        self.saved = saved
        self.x = x
        self.P = P
        return self.results


if __name__ == '__main__':
    file_path = "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/mid-accuracy"
    imu_params = {
        "G_CONST": 9.8,
        # 随机游走 (PSD)
        "ARW": 2.0e-3 * D2R / 60,  # rad/sqrt(s)
        "VRW": 0.01,  # m/s^2/sqrt(s)，根据加速度计噪声设
        # 传感器零偏
        "gyro_bias": 0.0,
        "accel_bias": 0.0,
        "sigma_bg": 0.1 * D2R / 3600,  # rad/s/sqrt(s) 零偏漂移
        "sigma_ba": 1e-3,  # m/s^2/sqrt(s)
        # 比例因子
        "gyro_scale": 1.0,
        "accel_scale": 1.0,
        "sigma_sg": 0.001,
        "sigma_sa": 0.001,
        # 一阶马尔科夫过程相关时间
        "Tgb": 100.0,
        "Tab": 100.0,
        "Tgs": 100.0,
        "Tas": 100.0,
        # 测量标准差
        "gyro_std": 0.0,
        "accel_std": 0.0,
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

    for i in range(len(gps_data)):
        col_mean = gps_data.mean(axis=0)
        # if i in range(10, 20):
        #     ratio = 2 * 1e5
        #     gps_data[i, 0] += np.random.randn() * col_mean[0] / ratio
        #     gps_data[i, 1] += np.random.randn() * col_mean[1] / ratio
        #     gps_data[i, 2] += np.random.randn() * col_mean[2] / ratio
        #     gps_data[i, 3] += np.random.randn() * col_mean[3] / ratio
        #     gps_data[i, 4] += np.random.randn() * col_mean[4] / ratio
        #     gps_data[i, 5] += np.random.randn() * col_mean[5] / ratio
        # elif i in range(30, 45):
        #     gps_data[i, 0] += np.random.randn() * col_mean[0] / ratio
        #     gps_data[i, 1] += np.random.randn() * col_mean[1] / ratio
        #     gps_data[i, 2] += np.random.randn() * col_mean[2] / ratio
        #     gps_data[i, 3] += np.random.randn() * col_mean[3] / ratio
        #     gps_data[i, 4] += np.random.randn() * col_mean[4] / ratio
        #     gps_data[i, 5] += np.random.randn() * col_mean[5] / ratio
    # 初始化 STEKF (保存 'lambda' 和 'P' 方便可视化分析)
    loose_stf = StrongTrackingLooseCouple(imu_params, imu_data, gps_data, init_vel_b, deg2rad(att_data[0][0:3]),
                                          save=['P', 'lambda'])

    ekf_states = loose_stf.run()
    for k in loose_stf.saved:
        if k == "lambda":
            continue
        value = loose_stf.saved[k]
        mat_list = np.array(value)  # shape = (T, 21, 21)
        T = mat_list.shape[0]

        plt.figure(figsize=(12, 8))

        for i in range(21):
            for j in range(21):
                plt.plot(range(T), mat_list[:, i, j], alpha=0.5)

        plt.title(f"Element-wise Change For {k}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    ins_result = loose_stf.ins.out_put
    ekf_pos = np.zeros_like(ins_result)
    for i in range(len(ekf_states)):
        s = ekf_states[i]

        # 注意：s[IDX_DR, 0] 是 EKF 对 δr (位置误差) 的估计
        # P_corrected = P_ins - δr_estimated
        ekf_pos[i] = ins_result[i] - s[IDX_DR, 0]
    ins_result = np.array(ins_result)
    plt.figure(figsize=(8, 8), dpi=120)

    plt.plot(ekf_pos[:, 1], ekf_pos[:, 0], label="AKF", linewidth=2)
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


    # 如果想看渐消因子的变化，可以加一个简单的作图：
    if 'lambda' in loose_stf.saved:
        plt.figure(figsize=(10, 4))
        plt.plot(loose_stf.saved['lambda'], color='red')
        plt.title("STEKF Fading Factor (λ) Over Time")
        plt.xlabel("GNSS Update Steps")
        plt.ylabel("λ value")
        plt.grid(True)
        plt.show()