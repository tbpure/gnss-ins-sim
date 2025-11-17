from KF.ekf_base import EKF
from ins.ins_algo import INS
import numpy as np

from utils.matrix_utils import skew

STATUS_DIMENSION = 21

IDX_DR = slice(0, 3) # 位置误差
IDX_DV = slice(3, 6) # 速度误差
IDX_PHI = slice(6, 9) # 姿态误差
IDX_BG = slice(9, 12) # 陀螺零偏误差
IDX_BA = slice(12, 15) # 加速度计零偏误差
IDX_SG = slice(15, 18) # 陀螺比例因子误差
IDX_SA = slice(18, 21) # 加速度计比例因子误差


def build_G(Cnb):
    # 过程噪声分布矩阵 G
    G = np.zeros((STATUS_DIMENSION, 18))
    G[IDX_DV, 0:3] = Cnb
    G[IDX_PHI, 3:6] = Cnb
    G[IDX_BG, 6:9] = np.eye(3)
    G[IDX_BA, 9:12] = np.eye(3)
    G[IDX_SG, 12:15] = np.eye(3)
    G[IDX_SA, 15:18] = np.eye(3)
    return G


class LooseCouple:
    def __init__(self, imu_params, gnss_params, imu_data, gnss_data, init_vel, init_euler, dt = 0.01):
        self.dt = 0.01
        self.imu_params = imu_params
        self.gnss_params = gnss_params
        self.imu_data = imu_data
        self.gnss_data = gnss_data
        self.ins = INS(imu_data, gnss_data, init_vel, init_euler)
        ekf = EKF(STATUS_DIMENSION)

        # 初始化状态和状态协方差矩阵
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


        # 零偏
        self.gyro_bias = imu_params["gyro_bias"]
        self.accel_bias = imu_params["accel_bias"]

        # 标准差
        self.gyro_std = imu_params["gyro_std"]
        self.accel_std = imu_params["accel_std"]

        # 比例因子
        self.gyro_scale = imu_params["gyro_scale"]
        self.accel_scale = imu_params["accel_scale"]


    def build_F(self, Cnb, acc, gyro, vn, pos, earth, imu_params):
        """
        构造完整 21x21 F 矩阵
        """

        F = np.zeros((STATUS_DIMENSION, STATUS_DIMENSION))

        # 地球/位置参数
        RM, RN = earth["RM"], earth["RN"]
        h, phi = pos[2], pos[0]  # lat, lon, h (注意：仿真中 pos[0] 是X, pos[1]是Y, 这里简化处理)
        vN, vE, vD = vn

        # 简化：假设 pos[0] (纬度) 和 omega_in_n 来自 earth 字典
        phi_lat = earth.get("lat", 0.0)  # 真实的纬度
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
        # todo: 重力项待修改
        Fvr[2, 2] = 2 * self.g / (RM + h)
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
        imu_params = self.imu_params.copy()
        gnss_params = self.gnss_params.copy()
        gnss_data = self.gnss_data.copy()
        ins_data = self.imu_data.copy()
        x = np.zeros((STATUS_DIMENSION, 1))
        P = np.eye(STATUS_DIMENSION)
        P[IDX_DR, IDX_DR] *= 1.0 ** 2
        P[IDX_DV, IDX_DV] *= 0.1 ** 2
        P[IDX_PHI, IDX_PHI] *= np.deg2rad(0.1) ** 2
        P[IDX_BG, IDX_BG] *= np.deg2rad(0.1 / 3600) ** 2
        # todo
        P[IDX_BA, IDX_BA] *= (1e-3 * self.imu_params["G_CONST"]) ** 2
        imu_params_q = self.imu_params.copy()
        imu_params_q["ARW"] = self.imu_params["ARW"] / np.sqrt(1 / 0.01)  # 假设 ARW/VRW 单位是 /sqrt(s)
        imu_params_q["VRW"] = self.imu_params["VRW"] / np.sqrt(1 / 0.01)

        # 修正：假设 ARW/VRW 单位已经是 (m/s^2)/sqrt(Hz) 或 (rad/s)/sqrt(Hz)
        # 那么 Q = sigma^2 / dt。 而连续时间 q = sigma^2 * dt。
        # 讲义中 ARW 是 sqrt(q_a * dt)，所以 q_a = ARW^2 / dt
        # 不，ARW/VRW 本身就是 sqrt(PSD)
        # q[0:3, 0:3] = (imu_params["ARW"] ** 2) * eye(3) # PSD = (m/s^2/sqrt(Hz))^2 = (m/s^2)^2/Hz
        q = self.build_q(imu_params_q)

        R_pos = np.eye(3) * 1.5 ** 2  # GNSS position variance (m^2)
        R_vel = np.eye(3) * 0.5 ** 2  # GNSS velocity variance (m^2/s^2)

        results = []
        gnss_index = 0

        for k, imu in enumerate(ins_data):
            dt = imu["dt"]

            # EKF 使用的输入：来自漂移的 INS 和带噪声的 IMU 读数
            acc = imu["acc_measured"]  # 带噪声的比力
            gyro = imu["gyro_measured"]  # 带噪声的角速度
            Cnb = imu["Cnb_ins"]  # 漂移的姿态
            vn = imu["vel_ins"]  # 漂移的速度
            pos = imu["pos_ins"]  # 漂移的位置
            earth = imu["earth"]

            # =====================================
            # ① 预测：IMU 100Hz 更新
            # =====================================
            F = self.build_F(Cnb, acc, gyro, vn, pos, earth, imu_params)
            G = build_G(Cnb)

            Phi = np.eye(STATUS_DIMENSION) + F * dt

            # 离散化 Qk (近似)：Qk = Phi * G * q * G.T * Phi.T * dt
            # 修正：Qk = G * q * G.T * dt (一阶近似)
            Qk = (G @ q @ G.T) * dt

            # 状态更新
            x = Phi @ x
            P = Phi @ P @ Phi.T + Qk

            # =====================================
            # ② 若此时有 GNSS 数据则进行更新
            # =====================================
            t = imu["timestamp"]
            if gnss_index < len(gnss_data) and abs(gnss_data[gnss_index]["timestamp"] - t) < dt / 2.0:
                gnss = gnss_data[gnss_index]
                gnss_index += 1

                # -----------------------
                # GNSS 位置观测
                # -----------------------
                lb = imu.get("lever_arm", np.zeros(3))
                H_pos = np.zeros((3, STATUS_DIMENSION))
                H_pos[:, IDX_DR] = np.eye(3)
                H_pos[:, IDX_PHI] = skew(Cnb @ lb)

                # 观测 z = INS 预测 - GNSS 测量
                z_pos = (imu["pos_ins"] - gnss["pos"]).reshape(3, 1)

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

                z_vel = (imu["vel_ins"] - gnss["vel"]).reshape(3, 1)
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

