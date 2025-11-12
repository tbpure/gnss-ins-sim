import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 参数设定
# -----------------------------------
dt_ins = 0.01     # INS采样间隔 (100Hz)
dt_gnss = 1.0     # GNSS采样间隔 (1Hz)
steps = int(200 / dt_ins)  # 总步数
gnss_interval = int(dt_gnss / dt_ins)

# 状态量 [δr, δv] (简化为6维)
n_state = 6
x = np.zeros((n_state, 1))
P = np.eye(n_state) * 1e-2

Phi = np.eye(n_state)
Phi[0:3, 3:6] = np.eye(3) * dt_ins
Q = np.eye(n_state) * 1e-5
R_gnss = np.eye(3) * 4.0  # GNSS位置观测噪声

# -----------------------------------
# 模拟轨迹
# -----------------------------------
true_pos = np.zeros((steps, 3))
true_vel = np.zeros((steps, 3))
ins_pos = np.zeros((steps, 3))
gnss_pos = np.zeros((steps, 3))

v_true = np.array([1.0, 0.5, 0])  # 匀速运动

for k in range(1, steps):
    true_pos[k] = true_pos[k - 1] + v_true * dt_ins
    true_vel[k] = v_true
    ins_pos[k] = ins_pos[k - 1] + (v_true + np.random.randn(3) * 0.02) * dt_ins
    if k % gnss_interval == 0:
        gnss_pos[k] = true_pos[k] + np.random.randn(3) * 2.0
    else:
        gnss_pos[k] = np.nan  # 无GNSS观测

# -----------------------------------
# 滤波主循环
# -----------------------------------
est_pos = np.zeros((steps, 3))
est_vel = np.zeros((steps, 3))

for k in range(1, steps):
    # 1️⃣ 时间更新（每次INS数据来时执行）
    x = Phi @ x
    P = Phi @ P @ Phi.T + Q

    # 2️⃣ 若此刻有GNSS观测，则执行量测更新
    if not np.isnan(gnss_pos[k, 0]):
        H = np.zeros((3, n_state))
        H[:, 0:3] = np.eye(3)
        z = (ins_pos[k] - gnss_pos[k]).reshape(3, 1)
        y = z - H @ x
        S = H @ P @ H.T + R_gnss
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(n_state) - K @ H) @ P

    # 3️⃣ 输出修正后的估计
    est_pos[k] = ins_pos[k] - x[0:3, 0]
    est_vel[k] = v_true - x[3:6, 0]

# -----------------------------------
# 可视化
# -----------------------------------
plt.figure(figsize=(8, 5))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'k-', label='True')
plt.plot(ins_pos[:, 0], ins_pos[:, 1], 'r--', label='INS Only')
plt.plot(est_pos[:, 0], est_pos[:, 1], 'b-', label='INS+GNSS EKF')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.legend()
plt.title('GNSS(1Hz) / INS(100Hz) 松组合频率对齐示例')
plt.show()