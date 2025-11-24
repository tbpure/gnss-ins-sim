# gpt_demo_fixed_v2.py
import numpy as np
import matplotlib.pyplot as plt

from utils.matrix_utils import skew

# matplotlib.use("TkAgg")

# ============================================================
# 常量
# ============================================================
G_CONST = 9.8  # 简化重力
R_E = 6378137.0  # 地球半径
OMEGA_E = 7.292115e-5  # 地球自转角速度 rad/s


# ============================================================
# 工具函数
# ============================================================

def eye(n):
    return np.eye(n)


def euler_to_dcm(roll, pitch, yaw):
    """ 欧拉角转方向余弦矩阵 (n系到b系) """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # ZYX 顺序: C_n^b
    C = np.array([
        [cp * cy, cp * sy, -sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
    ])
    return C


# ============================================================
# 21 维状态：δx = [δr, δv, ϕ, bg, ba, sg, sa]
# ============================================================

STATE_SIZE = 21

IDX_DR = slice(0, 3) # 位置误差
IDX_DV = slice(3, 6) # 速度误差
IDX_PHI = slice(6, 9) # 姿态误差
IDX_BG = slice(9, 12) # 陀螺零偏误差
IDX_BA = slice(12, 15) # 加速度计零偏误差
IDX_SG = slice(15, 18) # 陀螺比例因子误差
IDX_SA = slice(18, 21) # 加速度计比例因子误差


# ============================================================
# F 矩阵构建（补充科里奥利和重力项）
# ============================================================

def build_F(Cnb, acc, gyro, vn, pos, earth, imu_params):
    """
    构造完整 21x21 F 矩阵
    acc：载体系
    gyro：载体系
    vn：导航系
    pos：导航系
    """

    F = np.zeros((STATE_SIZE, STATE_SIZE))

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

    F[IDX_DR, IDX_DR] = Frr
    F[IDX_DR, IDX_DV] = eye(3)

    # ----------------------------------------
    # F_vr, F_vv (速度误差)
    # ----------------------------------------
    # F_vr (重力异常和牵连加速度项)
    Fvr = np.zeros((3, 3))
    # 简化重力模型 (垂向)
    Fvr[2, 2] = 2 * G_CONST / (RM + h)
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
    F[IDX_BG, IDX_BG] = -(1.0 / imu_params["Tgb"]) * eye(3)
    F[IDX_BA, IDX_BA] = -(1.0 / imu_params["Tab"]) * eye(3)
    F[IDX_SG, IDX_SG] = -(1.0 / imu_params["Tgs"]) * eye(3)
    F[IDX_SA, IDX_SA] = -(1.0 / imu_params["Tas"]) * eye(3)

    return F


# ============================================================
# G 与 Q（过程噪声）
# ============================================================

def build_G(Cnb):
    G = np.zeros((STATE_SIZE, 18))
    G[IDX_DV, 0:3] = Cnb
    G[IDX_PHI, 3:6] = Cnb
    G[IDX_BG, 6:9] = eye(3)
    G[IDX_BA, 9:12] = eye(3)
    G[IDX_SG, 12:15] = eye(3)
    G[IDX_SA, 15:18] = eye(3)
    return G


def build_q(imu_params):
    q = np.zeros((18, 18))
    q[0:3, 0:3] = (imu_params["ARW"] ** 2) * eye(3)
    q[3:6, 3:6] = (imu_params["VRW"] ** 2) * eye(3)
    q[6:9, 6:9] = 2.0 * (imu_params["sigma_bg"] ** 2) / imu_params["Tgb"] * eye(3)
    q[9:12, 9:12] = 2.0 * (imu_params["sigma_ba"] ** 2) / imu_params["Tab"] * eye(3)
    q[12:15, 12:15] = 2.0 * (imu_params["sigma_sg"] ** 2) / imu_params["Tgs"] * eye(3)
    q[15:18, 15:18] = 2.0 * (imu_params["sigma_sa"] ** 2) / imu_params["Tas"] * eye(3)
    return q


# ============================================================
# EKF 主循环
# ============================================================

def ekf_gnss_ins(ins_data_stream, gnss_data):
    """
    ins_data_stream: 包含 *漂移的* INS 解算结果和 *带噪声的* IMU 读数
    gnss_data: 稀疏的 GNSS 观测
    """

    # 初始化
    x = np.zeros((STATE_SIZE, 1))
    P = eye(STATE_SIZE)
    P[IDX_DR, IDX_DR] *= 1.0 ** 2
    P[IDX_DV, IDX_DV] *= 0.1 ** 2
    P[IDX_PHI, IDX_PHI] *= np.deg2rad(0.1) ** 2
    P[IDX_BG, IDX_BG] *= np.deg2rad(0.1 / 3600) ** 2
    P[IDX_BA, IDX_BA] *= (1e-3 * G_CONST) ** 2

    # EKF 使用的 IMU 噪声参数 (模型)
    imu_params = {
        "ARW": 0.01,  # 角度随机游走(m/s^2) / sqrt(Hz)
        "VRW": 0.001,  # 加速度随机游走(rad/s) / sqrt(Hz)
        # t表示马尔可夫过程的时间常数
        # '''
        # t表示马尔可夫过程的时间常数
        # bg为陀螺仪零漂
        # ba为加速度计零漂
        # sg为陀螺仪比例因子
        # sa为加速度计比例因子
        # '''

        "Tgb": 3600.0, "Tab": 3600.0, "Tgs": 3600.0, "Tas": 3600.0,
        "sigma_bg": np.deg2rad(0.05 / 3600),  # 0.05 deg/hr
        "sigma_ba": 1e-4 * G_CONST,  # 100 ug
        "sigma_sg": 0.0,
        "sigma_sa": 0.0
    }
    # 转换为 PSD (ARW/VRW 单位是 /sqrt(Hz)，q 中需要 PSD)
    # Note: ARW/VRW in build_q are already PSDs if units are e.g. m/s/sqrt(hz)
    # 修正：q 中使用的是功率谱密度，ARW/VRW 的单位已经是 /sqrt(Hz)，所以平方是对的
    imu_params_q = imu_params.copy()
    imu_params_q["ARW"] = imu_params["ARW"] / np.sqrt(1 / 0.01)  # 假设 ARW/VRW 单位是 /sqrt(s)
    imu_params_q["VRW"] = imu_params["VRW"] / np.sqrt(1 / 0.01)

    # 修正：假设 ARW/VRW 单位已经是 (m/s^2)/sqrt(Hz) 或 (rad/s)/sqrt(Hz)
    # 那么 Q = sigma^2 / dt。 而连续时间 q = sigma^2 * dt。
    # 讲义中 ARW 是 sqrt(q_a * dt)，所以 q_a = ARW^2 / dt
    # 不，ARW/VRW 本身就是 sqrt(PSD)
    # q[0:3, 0:3] = (imu_params["ARW"] ** 2) * eye(3) # PSD = (m/s^2/sqrt(Hz))^2 = (m/s^2)^2/Hz
    q = build_q(imu_params)

    R_pos = eye(3) * 1.5 ** 2  # GNSS position variance (m^2)
    R_vel = eye(3) * 0.5 ** 2  # GNSS velocity variance (m^2/s^2)

    results = []
    gnss_index = 0

    for k, imu in enumerate(ins_data_stream):
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
        F = build_F(Cnb, acc, gyro, vn, pos, earth, imu_params)
        G = build_G(Cnb)

        Phi = eye(STATE_SIZE) + F * dt

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
            H_pos = np.zeros((3, STATE_SIZE))
            H_pos[:, IDX_DR] = eye(3)
            H_pos[:, IDX_PHI] = skew(Cnb @ lb)

            # 观测 z = INS 预测 - GNSS 测量
            z_pos = (imu["pos_ins"] - gnss["pos"]).reshape(3, 1)

            # 创新 y = z - H * x_pred
            y = z_pos - H_pos @ x
            S = H_pos @ P @ H_pos.T + R_pos
            K = P @ H_pos.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (eye(STATE_SIZE) - K @ H_pos) @ P

            # -----------------------
            # GNSS 速度观测
            # -----------------------
            H_vel = np.zeros((3, STATE_SIZE))
            H_vel[:, IDX_DV] = eye(3)
            term = Cnb @ np.cross(lb, gyro)
            H_vel[:, IDX_PHI] = -skew(term)

            z_vel = (imu["vel_ins"] - gnss["vel"]).reshape(3, 1)
            y = z_vel - H_vel @ x
            S = H_vel @ P @ H_vel.T + R_vel
            K = P @ H_vel.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (eye(STATE_SIZE) - K @ H_vel) @ P

            # --- 状态反馈 (可选，若 INS 循环在外部) ---
            # 此处 EKF 是外部运行的，我们只保存误差状态 x
            # 在一个真实系统中，x 会被反馈回 INS 循环以修正 ins_pos, ins_vel

        # 保存结果（每个 IMU 时刻都保存）
        results.append(x.copy())

    return results


# ============================================================
# 绘图函数
# ============================================================

def plot_trajectory(true_pos, ins_pos, ekf_pos, gnss_pos=None):
    plt.figure(figsize=(9, 6))

    plt.plot(true_pos[:, 0], true_pos[:, 1], 'k-', linewidth=3, label='True')
    plt.plot(ins_pos[:, 0], ins_pos[:, 1], 'r--', alpha=0.8, label='INS Only (Drifting)')
    plt.plot(ekf_pos[:, 0], ekf_pos[:, 1], 'b-', linewidth=2, label='EKF Fused')

    if gnss_pos is not None:
        plt.scatter(gnss_pos[:, 0], gnss_pos[:, 1], c='g', s=20, label='GNSS (1Hz)', zorder=5)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory Comparison (True / INS / EKF)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    # plt.tight_layout()
    plt.show()


# ============================================================
# 仿真主程序 (已重写)
# ============================================================


def gpt_test_data():
    # 仿真参数
    T = 120.0
    dt_imu = 0.01
    N = int(T / dt_imu)

    # 仿真地球参数
    LATITUDE_RAD = np.deg2rad(45.0)  # 假设纬度
    g_n = np.array([0, 0, -G_CONST])  # n系重力 (北东地 NED 坐标系)
    omega_ie_n = np.array([0, OMEGA_E * np.cos(LATITUDE_RAD), OMEGA_E * np.sin(LATITUDE_RAD)])

    # ----------------------------------------------------
    # 1. 生成地面真值 (Ground Truth)
    # ----------------------------------------------------
    true_pos = np.zeros((N, 3))
    true_vel = np.zeros((N, 3))
    true_Cnb = np.zeros((N, 3, 3))
    true_acc_n = np.zeros((N, 3))
    true_wb_n = np.zeros((N, 3))  # n系下的角速度 (omega_nb_n)

    # 轨迹：前半段直行，后半段匀角速度转弯
    yaw = 0.0
    turn_rate = np.deg2rad(0.5)  # 0.5 deg/s

    for k in range(N):
        t = k * dt_imu
        if t < T / 3:
            vx, vy = 3.0, 0.0
            ax, ay = 0.0, 0.0
            yaw_rate = 0.0
        elif t < (T * 2 / 3):
            # 匀速转弯
            yaw_rate = turn_rate
            yaw += yaw_rate * dt_imu
            vx = 3.0 * np.cos(yaw)
            vy = 3.0 * np.sin(yaw)
            # 计算向心加速度 a = v * w
            ax = -vy * yaw_rate
            ay = vx * yaw_rate
        else:
            # 再次直行
            yaw_rate = 0.0
            vx = 3.0 * np.cos(yaw)
            vy = 3.0 * np.sin(yaw)
            ax, ay = 0.0, 0.0

        true_vel[k] = [vx, vy, 0.0]
        true_acc_n[k] = [ax, ay, 0.0]
        true_wb_n[k] = [0, 0, yaw_rate]

        if k > 0:
            true_pos[k] = true_pos[k - 1] + true_vel[k] * dt_imu

        true_Cnb[k] = euler_to_dcm(0.0, 0.0, yaw).T  # NED -> Body, Cnb (e.g., ZYX yaw-pitch-roll)
        # 注意：euler_to_dcm 假设是 n->b, 需确认
        # 假设 euler_to_dcm(0,0,yaw) 是 C_b^n (b到n)
        # 那么 C_n^b (n到b) 应该是 C_b^n.T
        # 我们的 Cnb 是 n->b (IMU坐标系)

        # 修正：使用 euler_to_dcm (0,0,yaw) 作为 C_n^b
        # C_n^b = euler_to_dcm(0.0, 0.0, yaw) # Cnb = ZYX(yaw, pitch, roll)

        # 修正：原 euler_to_dcm 是 C_b^n (b 到 n)，坐标系 (x,y,z) 对应 (roll, pitch, yaw)
        # 我们用 NED (北东地)，yaw 是绕 D 轴转
        # 假设使用 北东地 (NED) 坐标系， yaw 是绕 D (Z) 轴
        # C_n^b (n系到b系)
        roll, pitch = 0.0, 0.0
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # ZYX (Yaw-Pitch-Roll) C_n^b
        Cnb = np.array([
            [cp * cy, cp * sy, -sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
        ])
        true_Cnb[k] = Cnb

    # ----------------------------------------------------
    # 2. 生成带噪声的 IMU 传感器读数
    # ----------------------------------------------------
    # 传感器噪声参数 (用于生成数据)
    gyro_bias = np.deg2rad(np.array([0.5, -0.3, 0.2]) / 3600)  # 0.5 deg/hr 零偏
    acc_bias = np.array([-0.5, 0.8, -0.2]) * 1e-3 * G_CONST  # 1 mG 零偏

    ARW_std = 0.01 / np.sqrt(dt_imu)  # (m/s^2)/sqrt(Hz) -> m/s^2
    VRW_std = 0.001 / np.sqrt(dt_imu)  # (rad/s)/sqrt(Hz) -> rad/s

    imu_sensor_readings = []

    for k in range(N):
        Cnb = true_Cnb[k]
        Cbn = Cnb.T

        # --- 真实比力 ---
        # fb = C_n^b * (a_n - g_n)
        fb_true = Cnb @ (true_acc_n[k] - g_n)

        # --- 真实角速度 ---
        # omega_ib_b = omega_in_b + omega_nb_b
        # omega_in_b = C_n^b * omega_in_n (omega_in_n = omega_ie_n + omega_en_n)
        # (简化：忽略 omega_en_n)
        omega_in_b = Cnb @ omega_ie_n
        # omega_nb_b = C_n^b * omega_nb_n
        omega_nb_b = Cnb @ true_wb_n[k]

        wb_true = omega_in_b + omega_nb_b

        # --- 添加噪声和零偏 ---
        acc_noise = np.random.randn(3) * ARW_std
        gyro_noise = np.random.randn(3) * VRW_std

        acc_measured = fb_true + acc_bias + acc_noise
        gyro_measured = wb_true + gyro_bias + gyro_noise

        imu_sensor_readings.append({
            "acc_measured": acc_measured,
            "gyro_measured": gyro_measured
        })

    # ----------------------------------------------------
    # 3. 模拟 "INS Only" 纯惯导解算 (产生漂移)
    # ----------------------------------------------------
    ins_pos = np.zeros_like(true_pos)
    ins_vel = np.zeros_like(true_vel)
    ins_Cnb = np.zeros_like(true_Cnb)

    # 初始状态 (假设有很小的初始误差)
    ins_pos[0] = true_pos[0] + np.array([0.1, -0.1, 0.0])
    ins_vel[0] = true_vel[0] + np.array([0.01, -0.01, 0.0])
    ins_Cnb[0] = true_Cnb[0]  # 假设初始对准完美

    # EKF 输入流 (包含漂移的 INS 和 IMU 读数)
    ekf_input_stream = []

    for k in range(N):
        # 准备 EKF 输入数据
        ekf_input_stream.append({
            "timestamp": k * dt_imu,
            "dt": dt_imu,
            # IMU 原始读数
            "acc_measured": imu_sensor_readings[k]["acc_measured"],
            "gyro_measured": imu_sensor_readings[k]["gyro_measured"],
            # 漂移的 INS 状态 (来自上一步 k-1)
            "Cnb_ins": ins_Cnb[k - 1] if k > 0 else ins_Cnb[0],
            "vel_ins": ins_vel[k - 1] if k > 0 else ins_vel[0],
            "pos_ins": ins_pos[k - 1] if k > 0 else ins_pos[0],
            # 地球参数
            "earth": {
                "RM": R_E, "RN": R_E,
                "omega_in_n": omega_ie_n,
                "lat": LATITUDE_RAD
            },
            "lever_arm": np.zeros(3),
        })

        if k == N - 1: break  # 最后一步不需要更新

        # --- INS 积分 (k -> k+1) ---

        # 使用 k 时刻的 IMU 读数 和 k 时刻的 INS 状态
        fb = imu_sensor_readings[k]["acc_measured"]
        wb = imu_sensor_readings[k]["gyro_measured"]

        Cnb_k = ins_Cnb[k]
        vel_k = ins_vel[k]
        pos_k = ins_pos[k]

        # (1) 姿态更新 (C_n^b)
        # 忽略 omega_in_n 的旋转 (让它漂移，EKF 会估计这个)
        # dCnb = Cnb_k @ skew(wb) * dt_imu
        # Cnb_k_plus_1 = Cnb_k + dCnb

        # (更精确的姿态更新)
        d_theta = wb * dt_imu
        Cnb_k_plus_1 = Cnb_k @ (eye(3) + skew(d_theta))
        # (还应减去 Cnb * omega_in_n * dt ... 但我们让它漂移)

        ins_Cnb[k + 1] = Cnb_k_plus_1

        # (2) 速度更新 (n系)
        # f_n = C_b^n * f_b = Cnb.T * fb
        f_n = Cnb_k.T @ fb

        # a_n = f_n + g_n - (2*omega_ie + omega_en) x v_n
        # (简化：INS 循环不补偿科里奥利力，让它漂移)
        a_n = f_n + g_n

        ins_vel[k + 1] = vel_k + a_n * dt_imu

        # (3) 位置更新
        ins_pos[k + 1] = pos_k + ins_vel[k + 1] * dt_imu

    # ----------------------------------------------------
    # 4. 生成稀疏的 GNSS 测量
    # ----------------------------------------------------
    gnss_data = []
    gnss_sparse = []
    gnss_noise_pos_std = 1.0  # 1.0 米
    gnss_noise_vel_std = 0.3  # 0.3 m/s

    for k in range(0, N, int(1.0 / dt_imu)):
        pos_noise = np.random.randn(3) * gnss_noise_pos_std
        vel_noise = np.random.randn(3) * gnss_noise_vel_std

        gnss_data.append({
            "timestamp": k * dt_imu,
            "pos": true_pos[k] + pos_noise,
            "vel": true_vel[k] + vel_noise
        })
        gnss_sparse.append(true_pos[k] + pos_noise)
    gnss_sparse = np.array(gnss_sparse)
    return gnss_data, ekf_input_stream, gnss_sparse


if __name__ == "__main__":
    # 仿真参数
    T = 120.0
    dt_imu = 0.01
    N = int(T / dt_imu)

    # 仿真地球参数
    LATITUDE_RAD = np.deg2rad(45.0)  # 假设纬度
    g_n = np.array([0, 0, -G_CONST])  # n系重力 (北东地 NED 坐标系)
    omega_ie_n = np.array([0, OMEGA_E * np.cos(LATITUDE_RAD), OMEGA_E * np.sin(LATITUDE_RAD)])

    # ----------------------------------------------------
    # 1. 生成地面真值 (Ground Truth)
    # ----------------------------------------------------
    true_pos = np.zeros((N, 3))
    true_vel = np.zeros((N, 3))
    true_Cnb = np.zeros((N, 3, 3))
    true_acc_n = np.zeros((N, 3))
    true_wb_n = np.zeros((N, 3))  # n系下的角速度 (omega_nb_n)

    # 轨迹：前半段直行，后半段匀角速度转弯
    yaw = 0.0
    turn_rate = np.deg2rad(0.5)  # 0.5 deg/s

    for k in range(N):
        t = k * dt_imu
        if t < T / 3:
            vx, vy = 3.0, 0.0
            ax, ay = 0.0, 0.0
            yaw_rate = 0.0
        elif t < (T * 2 / 3):
            # 匀速转弯
            yaw_rate = turn_rate
            yaw += yaw_rate * dt_imu
            vx = 3.0 * np.cos(yaw)
            vy = 3.0 * np.sin(yaw)
            # 计算向心加速度 a = v * w
            ax = -vy * yaw_rate
            ay = vx * yaw_rate
        else:
            # 再次直行
            yaw_rate = 0.0
            vx = 3.0 * np.cos(yaw)
            vy = 3.0 * np.sin(yaw)
            ax, ay = 0.0, 0.0

        true_vel[k] = [vx, vy, 0.0]
        true_acc_n[k] = [ax, ay, 0.0]
        true_wb_n[k] = [0, 0, yaw_rate]

        if k > 0:
            true_pos[k] = true_pos[k - 1] + true_vel[k] * dt_imu

        true_Cnb[k] = euler_to_dcm(0.0, 0.0, yaw).T  # NED -> Body, Cnb (e.g., ZYX yaw-pitch-roll)
        # 注意：euler_to_dcm 假设是 n->b, 需确认
        # 假设 euler_to_dcm(0,0,yaw) 是 C_b^n (b到n)
        # 那么 C_n^b (n到b) 应该是 C_b^n.T
        # 我们的 Cnb 是 n->b (IMU坐标系)

        # 修正：使用 euler_to_dcm (0,0,yaw) 作为 C_n^b
        # C_n^b = euler_to_dcm(0.0, 0.0, yaw) # Cnb = ZYX(yaw, pitch, roll)

        # 修正：原 euler_to_dcm 是 C_b^n (b 到 n)，坐标系 (x,y,z) 对应 (roll, pitch, yaw)
        # 我们用 NED (北东地)，yaw 是绕 D 轴转
        # 假设使用 北东地 (NED) 坐标系， yaw 是绕 D (Z) 轴
        # C_n^b (n系到b系)
        roll, pitch = 0.0, 0.0
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # ZYX (Yaw-Pitch-Roll) C_n^b
        Cnb = np.array([
            [cp * cy, cp * sy, -sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
        ])
        true_Cnb[k] = Cnb

    # ----------------------------------------------------
    # 2. 生成带噪声的 IMU 传感器读数
    # ----------------------------------------------------
    # 传感器噪声参数 (用于生成数据)
    gyro_bias = np.deg2rad(np.array([0.5, -0.3, 0.2]) / 3600)  # 0.5 deg/hr 零偏
    acc_bias = np.array([-0.5, 0.8, -0.2]) * 1e-3 * G_CONST  # 1 mG 零偏

    ARW_std = 0.01 / np.sqrt(dt_imu)  # (m/s^2)/sqrt(Hz) -> m/s^2
    VRW_std = 0.001 / np.sqrt(dt_imu)  # (rad/s)/sqrt(Hz) -> rad/s

    imu_sensor_readings = []

    for k in range(N):
        Cnb = true_Cnb[k]
        Cbn = Cnb.T

        # --- 真实比力 ---
        # fb = C_n^b * (a_n - g_n)
        fb_true = Cnb @ (true_acc_n[k] - g_n)

        # --- 真实角速度 ---
        # omega_ib_b = omega_in_b + omega_nb_b
        # omega_in_b = C_n^b * omega_in_n (omega_in_n = omega_ie_n + omega_en_n)
        # (简化：忽略 omega_en_n)
        omega_in_b = Cnb @ omega_ie_n
        # omega_nb_b = C_n^b * omega_nb_n
        omega_nb_b = Cnb @ true_wb_n[k]

        wb_true = omega_in_b + omega_nb_b

        # --- 添加噪声和零偏 ---
        acc_noise = np.random.randn(3) * ARW_std
        gyro_noise = np.random.randn(3) * VRW_std

        acc_measured = fb_true + acc_bias + acc_noise
        gyro_measured = wb_true + gyro_bias + gyro_noise

        imu_sensor_readings.append({
            "acc_measured": acc_measured,
            "gyro_measured": gyro_measured
        })

    # ----------------------------------------------------
    # 3. 模拟 "INS Only" 纯惯导解算 (产生漂移)
    # ----------------------------------------------------
    ins_pos = np.zeros_like(true_pos)
    ins_vel = np.zeros_like(true_vel)
    ins_Cnb = np.zeros_like(true_Cnb)

    # 初始状态 (假设有很小的初始误差)
    ins_pos[0] = true_pos[0] + np.array([0.1, -0.1, 0.0])
    ins_vel[0] = true_vel[0] + np.array([0.01, -0.01, 0.0])
    ins_Cnb[0] = true_Cnb[0]  # 假设初始对准完美

    # EKF 输入流 (包含漂移的 INS 和 IMU 读数)
    ekf_input_stream = []

    for k in range(N):
        # 准备 EKF 输入数据
        ekf_input_stream.append({
            "timestamp": k * dt_imu,
            "dt": dt_imu,
            # IMU 原始读数
            "acc_measured": imu_sensor_readings[k]["acc_measured"],
            "gyro_measured": imu_sensor_readings[k]["gyro_measured"],
            # 漂移的 INS 状态 (来自上一步 k-1)
            "Cnb_ins": ins_Cnb[k - 1] if k > 0 else ins_Cnb[0],
            "vel_ins": ins_vel[k - 1] if k > 0 else ins_vel[0],
            "pos_ins": ins_pos[k - 1] if k > 0 else ins_pos[0],
            # 地球参数
            "earth": {
                "RM": R_E, "RN": R_E,
                "omega_in_n": omega_ie_n,
                "lat": LATITUDE_RAD
            },
            "lever_arm": np.zeros(3),
        })

        if k == N - 1: break  # 最后一步不需要更新

        # --- INS 积分 (k -> k+1) ---

        # 使用 k 时刻的 IMU 读数 和 k 时刻的 INS 状态
        fb = imu_sensor_readings[k]["acc_measured"]
        wb = imu_sensor_readings[k]["gyro_measured"]

        Cnb_k = ins_Cnb[k]
        vel_k = ins_vel[k]
        pos_k = ins_pos[k]

        # (1) 姿态更新 (C_n^b)
        # 忽略 omega_in_n 的旋转 (让它漂移，EKF 会估计这个)
        # dCnb = Cnb_k @ skew(wb) * dt_imu
        # Cnb_k_plus_1 = Cnb_k + dCnb

        # (更精确的姿态更新)
        d_theta = wb * dt_imu
        Cnb_k_plus_1 = Cnb_k @ (eye(3) + skew(d_theta))
        # (还应减去 Cnb * omega_in_n * dt ... 但我们让它漂移)

        ins_Cnb[k + 1] = Cnb_k_plus_1

        # (2) 速度更新 (n系)
        # f_n = C_b^n * f_b = Cnb.T * fb
        f_n = Cnb_k.T @ fb

        # a_n = f_n + g_n - (2*omega_ie + omega_en) x v_n
        # (简化：INS 循环不补偿科里奥利力，让它漂移)
        a_n = f_n + g_n

        ins_vel[k + 1] = vel_k + a_n * dt_imu

        # (3) 位置更新
        ins_pos[k + 1] = pos_k + ins_vel[k + 1] * dt_imu

    # ----------------------------------------------------
    # 4. 生成稀疏的 GNSS 测量
    # ----------------------------------------------------
    gnss_data = []
    gnss_sparse = []
    gnss_noise_pos_std = 1.0  # 1.0 米
    gnss_noise_vel_std = 0.3  # 0.3 m/s

    for k in range(0, N, int(1.0 / dt_imu)):
        pos_noise = np.random.randn(3) * gnss_noise_pos_std
        vel_noise = np.random.randn(3) * gnss_noise_vel_std

        gnss_data.append({
            "timestamp": k * dt_imu,
            "pos": true_pos[k] + pos_noise,
            "vel": true_vel[k] + vel_noise
        })
        gnss_sparse.append(true_pos[k] + pos_noise)
    gnss_sparse = np.array(gnss_sparse)

    # ----------------------------------------------------
    # 5. 运行 EKF
    # ----------------------------------------------------
    print("Running EKF...")
    # EKF 接收包含 INS 漂移 和 IMU 读数的数据流
    ekf_states = ekf_gnss_ins(ekf_input_stream, gnss_data)
    print("EKF Done.")

    # ----------------------------------------------------
    # 6. 修正轨迹
    # ----------------------------------------------------
    # EKF 的结果是误差状态 [δr, δv, ...]
    # 修正后的位置 = INS 漂移位置 - EKF 估计的位置误差(δr)

    ekf_pos = np.zeros_like(ins_pos)
    ekf_vel = np.zeros_like(ins_vel)

    if len(ekf_states) != N:
        print(f"Warning: EKF states length ({len(ekf_states)}) does not match N ({N})")

    for i in range(len(ekf_states)):
        s = ekf_states[i]

        # 注意：s[IDX_DR, 0] 是 EKF 对 δr (位置误差) 的估计
        # P_corrected = P_ins - δr_estimated
        ekf_pos[i] = ins_pos[i] - s[IDX_DR, 0]
        ekf_vel[i] = ins_vel[i] - s[IDX_DV, 0]

    # ----------------------------------------------------
    # 7. 绘图
    # ----------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(true_pos[:, 0], true_pos[:, 1], 'k-', linewidth=3, label='True')
    plt.plot(ins_pos[:, 0], ins_pos[:, 1], 'r--', alpha=0.8, label='INS Only (Drifting)')
    plt.plot(ekf_pos[:, 0], ekf_pos[:, 1], 'b-', linewidth=2, label='EKF Fused')

    if gnss_sparse is not None:
        plt.scatter(gnss_sparse[:, 0], gnss_sparse[:, 1], c='g', s=20, label='GNSS (1Hz)', zorder=5)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory Comparison (True / INS / EKF)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    # plt.tight_layout()
    plt.show()