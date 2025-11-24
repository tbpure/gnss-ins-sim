import numpy as np
import matplotlib.pyplot as plt

# 示例数据
N = 100
t = np.linspace(0, 2*np.pi, N)
true_pos = np.stack((np.cos(t), np.sin(t)), axis=1)           # 真实轨迹：圆
ins_pos = true_pos + 0.1*np.random.randn(N, 2)               # INS轨迹，加噪声
ekf_pos = true_pos + 0.02*np.random.randn(N, 2)              # EKF轨迹，加较小噪声

# 绘图
plt.figure(figsize=(9, 6))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'k-', linewidth=3, label='True')
plt.plot(ins_pos[:, 0], ins_pos[:, 1], 'r--', alpha=0.8, label='INS Only (Drifting)')
plt.plot(ekf_pos[:, 0], ekf_pos[:, 1], 'b-', linewidth=2, label='EKF')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Trajectory Comparison')
plt.legend()
plt.axis('equal')   # 坐标等比
plt.grid(True)
plt.show()