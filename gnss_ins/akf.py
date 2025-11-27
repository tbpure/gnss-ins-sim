import numpy as np
from matplotlib import pyplot as plt

from gnss_ins.loose_couple import calc_RN_RM, calc_omega_en_n, calc_omega_ie_n, calc_gravity
from ins.run_algo import D2R
from sim_data_gen.gnss_ins_sim.attitude import attitude
from sim_data_gen.gnss_ins_sim.geoparams import geoparams
from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path, get_att_data_from_path, \
    get_ref_data_from_path
from utils.matrix_utils import skew
from utils.unit_transfer import deg2rad

# 假定 STATUS_DIMENSION, IDX_* 常量与前文一致
STATUS_DIMENSION = 21
IDX_DR = slice(0, 3)
IDX_DV = slice(3, 6)
IDX_PHI = slice(6, 9)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)
IDX_SG = slice(15, 18)
IDX_SA = slice(18, 21)


class AdaptiveLooseCouple:
    """
    基于你原来 LooseCouple 的自适应卡尔曼滤波（AKF）实现。
    采用 innovation-based 自适应观测噪声估计（R_pos, R_vel）。
    """

    def __init__(self, imu_params, imu_data, gnss_data, init_vel, init_euler,
                 adapt_alpha=0.98, r_min_pos=0.1, r_min_vel=0.01, dt=0.01, save=None):
        """
        adapt_alpha: 创新协方差指数平滑因子（越接近1越慢变化）
        r_min_pos / r_min_vel: 估计出的 R 对角线下限（防止变成负或过小）
        dt: IMU 采样间隔
        """
        if save is None:
            self.save = []
        else:
            self.save = save
        self.imu_params = imu_params.copy()
        self.imu_data = imu_data.copy()
        self.gnss_data = gnss_data.copy()
        self.dt = dt

        # 初始化 INS （与你原来代码一致）
        from ins.ins_algo import INS  # 延迟导入以兼容你的工程结构
        self.ins = INS(imu_data, gnss_data, init_vel, init_euler)

        # 初始 EKF 状态（仅误差态）
        self.x = np.zeros((STATUS_DIMENSION, 1))
        self.P = np.eye(STATUS_DIMENSION)
        self.P[IDX_DR, IDX_DR] *= 1.0 ** 2
        self.P[IDX_DV, IDX_DV] *= 0.1 ** 2
        self.P[IDX_PHI, IDX_PHI] *= np.deg2rad(0.1) ** 2
        self.P[IDX_BG, IDX_BG] *= np.deg2rad(0.1 / 3600) ** 2
        self.P[IDX_BA, IDX_BA] *= (1e-3 * imu_params["G_CONST"]) ** 2

        # 观测噪声初值（可调整）
        self.R_pos = np.eye(3) * 1.5 ** 2
        self.R_vel = np.eye(3) * 0.5 ** 2

        # 自适应参数
        self.adapt_alpha = adapt_alpha
        self.r_min_pos = r_min_pos
        self.r_min_vel = r_min_vel

        # 创新协方差的指数平滑估计 (初始化以R为基准)
        self.Sy_pos = self.R_pos.copy()
        self.Sy_vel = self.R_vel.copy()

        # 其它状态拷贝（方便构建 F/G/q）
        self.gyro_bias = imu_params.get("gyro_bias", 0.0)
        self.accel_bias = imu_params.get("accel_bias", 0.0)
        self.gyro_std = imu_params.get("gyro_std", 0.0)
        self.accel_std = imu_params.get("accel_std", 0.0)

        # 保留一些数据以供外部检查
        self.saved = {}
        self.results = []

    # 将你之前的 build_F, build_G, build_q 等函数照抄到这里（或直接 import）
    # 为简洁起见，假设这些函数与 LooseCouple 的签名一致并在此类中实现
    def build_F(self, Cnb, acc, gyro, vn, ecef_pos, earth, imu_params):
        # 这里直接复制你原来的 build_F 实现（保持一致）
        F = np.zeros((STATUS_DIMENSION, STATUS_DIMENSION))
        lla_pos = geoparams.ecef2lla(ecef_pos)
        RM, RN = earth["RM"], earth["RN"]
        h, phi = lla_pos[2], lla_pos[0]
        vN, vE, vD = vn
        phi_lat = earth.get("lat", 0.0)
        omega_in_n = earth.get("omega_in_n", np.zeros(3))

        # Frr (位置-位置耦合)
        Frr = np.array([
            [-vD / (RM + h), 0, vN / (RM + h)],
            [vE * np.tan(phi_lat) / (RN + h), -(vD - vN * np.tan(phi_lat)) / (RN + h), vE / (RN + h)],
            [0, 0, 0]
        ])
        F[IDX_DR, IDX_DR] = Frr
        F[IDX_DR, IDX_DV] = np.eye(3)

        Fvr = np.zeros((3, 3))
        # 使用 self.g_n 作为近似重力（NED）
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

    # self-contained adaptive R update routine
    def adapt_R_from_innovation(self, y, H, P, R_est_container, Sy_container, adapt_alpha, r_min_diag):
        """
        Innovation-based adaptive R estimation (scalar/diag enforced).
        y: innovation vector (m x 1)
        H: measurement matrix (m x n)
        P: state covariance (n x n)
        R_est_container: reference to current R estimate (m x m)
        Sy_container: current innovation covariance estimate (m x m)
        adapt_alpha: smoothing factor for Sy update
        r_min_diag: minimum diag values (float or array)
        """
        # 更新瞬时创新二次型
        y = y.reshape(-1, 1)
        Sy_new = adapt_alpha * Sy_container + (1.0 - adapt_alpha) * (y @ y.T)
        # 理论上的 Sy = H P H^T + R  => R_est = Sy - H P H^T
        HPHT = H @ P @ H.T
        R_est = Sy_new - HPHT

        # 强制对角化，且下限（你可以改成更复杂的 PD 修正）
        R_diag = np.diag(R_est)
        # 若某些 R_diag 非正则或太小，则以下限替代
        R_diag_clamped = np.maximum(R_diag, r_min_diag)
        R_new = np.diag(R_diag_clamped)

        # 写回
        Sy_container[:, :] = Sy_new
        R_est_container[:, :] = R_new
        return R_new

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
            # 构建 earth 字典（可以直接 copy 你的 build_earth）
            RN = None
            RM = None
            # 这里调用你外部工具（或直接实现 build_earth），为节省篇幅假设已实现
            # 但我们需要 RN, RM, omega_in_n, g_n 等，简化写：
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

            # predict
            x = Phi @ x
            P = Phi @ P @ Phi.T + Qk

            # GNSS update (假设 GNSS rate = IMU_rate/100)
            if i % 100 == 0:
                gnss_index = int(i / 100)
                gnss = gnss_data[gnss_index][0:3]

                # ----- position update -----
                H_pos = np.zeros((3, STATUS_DIMENSION))
                H_pos[:, IDX_DR] = np.eye(3)
                lb = np.zeros(3)
                H_pos[:, IDX_PHI] = skew(Cnb @ lb)
                z_pos = (ins.position - gnss).reshape(3, 1)
                y_pos = z_pos - H_pos @ x
                S_pos = H_pos @ P @ H_pos.T + self.R_pos
                K_pos = P @ H_pos.T @ np.linalg.inv(S_pos)
                x = x + K_pos @ y_pos
                P = (np.eye(STATUS_DIMENSION) - K_pos @ H_pos) @ P

                # 自适应更新 R_pos（基于 innovation）
                # 这里用对角线下界 r_min_pos
                Rpos_new = self.adapt_R_from_innovation(y_pos, H_pos, P, self.R_pos, self.Sy_pos,
                                                        self.adapt_alpha, self.r_min_pos)

                # ----- velocity update -----
                H_vel = np.zeros((3, STATUS_DIMENSION))
                H_vel[:, IDX_DV] = np.eye(3)
                term = Cnb @ np.cross(lb, gyro)
                H_vel[:, IDX_PHI] = -skew(term)
                z_vel = (ins.vel - gnss_data[gnss_index][3:6]).reshape(3, 1)
                y_vel = z_vel - H_vel @ x
                S_vel = H_vel @ P @ H_vel.T + self.R_vel
                K_vel = P @ H_vel.T @ np.linalg.inv(S_vel)
                x = x + K_vel @ y_vel
                P = (np.eye(STATUS_DIMENSION) - K_vel @ H_vel) @ P

                Rvel_new = self.adapt_R_from_innovation(y_vel, H_vel, P, self.R_vel, self.Sy_vel,
                                                        self.adapt_alpha, self.r_min_vel)

                # 存储 P
                if 'P' in self.save:
                    saved.setdefault('P', []).append(P.copy())

            # 保存每步 x
            self.results.append(x.copy())

        self.saved = saved
        self.x = x
        self.P = P
        return self.results


if __name__ == "__main__":
    file_path = "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/default/2025-11-24-16-26-53"
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
    loose = AdaptiveLooseCouple(imu_params, imu_data, gps_data, init_vel_b, deg2rad(att_data[0][0:3]), save=['P'])


    ekf_states = loose.run()
    for k in loose.saved:
        value = loose.saved[k]
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
    ins_result = loose.ins.out_put
    ekf_pos = np.zeros_like(ins_result)
    for i in range(len(ekf_states)):
        s = ekf_states[i]

        # 注意：s[IDX_DR, 0] 是 EKF 对 δr (位置误差) 的估计
        # P_corrected = P_ins - δr_estimated
        ekf_pos[i] = ins_result[i] - s[IDX_DR, 0]
    ins_result = np.array(ins_result)
    plt.figure(figsize=(8, 8), dpi=120)

    plt.plot(ekf_pos[:, 1], ekf_pos[:, 0], label="AKF", linewidth=2)
    # plt.plot(ins_result[:, 1], ins_result[:, 0], label="INS", linewidth=2, alpha=0.8)
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