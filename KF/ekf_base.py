import numpy as np


class EKF:
    def __init__(self, n_states):
        """
        初始化 EKF
        :param n_states: 状态向量的维度 (n)，在你的例子中是 21
        """
        self.n = n_states

        # 状态向量 x (n x 1) - 默认为 0
        # 在 ES-EKF 中，这代表 *误差* 状态 δx
        self.x = np.zeros((self.n, 1))

        # 状态协方差矩阵 P (n x n)
        self.P = np.eye(self.n) * 0.01  # 用一个较小的值初始化

    def set_initial_state(self, x0, P0):
        """
        (可选) 设置一个非默认的初始状态和协方差
        """
        self.x = x0.reshape(self.n, 1)
        self.P = P0

    def predict(self, Phi, Qk):
        """
        EKF 预测步骤 (时间更新)

        :param Phi: 离散的状态转移矩阵 (n x n)
        :param Qk: 离散的过程噪声协方差矩阵 (n x n)
        """

        # 1. 状态预测: x_k|k-1 = Φ * x_k-1|k-1
        self.x = Phi @ self.x

        # 2. 协方差预测: P_k|k-1 = Φ * P_k-1|k-1 * Φ^T + Q_k
        self.P = Phi @ self.P @ Phi.T + Qk

        # (可选) 保持 P 矩阵的对称性，提高数值稳定性
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, y, H, R):
        """
        EKF 更新步骤 (测量更新)

        :param y: 测量新息 (z - h(x_pred)) (m x 1)
        :param H: 测量雅可比矩阵 (m x n)
        :param R: 测量噪声协方差矩阵 (m x m)
        """

        # 1. 计算新息协方差: S = H * P * H^T + R
        S = H @ self.P @ H.T + R

        # 2. 计算卡尔曼增益: K = P * H^T * S^-1
        #    使用 np.linalg.pinv (伪逆) 代替 np.linalg.inv (逆)
        #    可以提高在 S 接近奇异时（例如 R 很小）的数值稳定性
        K = self.P @ H.T @ np.linalg.pinv(S)

        # 3. 更新状态: x_k|k = x_k|k-1 + K * y
        self.x = self.x + K @ y

        # 4. 更新协方差: P_k|k = (I - K * H) * P_k|k-1
        #    使用 Joseph 形式的协方差更新，它更稳定且保证 P 保持对称
        I = np.eye(self.n)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # (可选) 再次确保对称性
        self.P = 0.5 * (self.P + self.P.T)

    def get_state(self):
        """
        获取当前的误差状态和协方差
        """
        return self.x, self.P

    def reset_state(self):
        """
        (专门用于 ES-EKF)
        当误差状态 δx 被反馈回主 INS 状态后，
        这个函数将误差状态向量重置为 0。
        协方差 P 保持不变。
        """
        self.x.fill(0.0)