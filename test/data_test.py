def test_q():
    import numpy as np
    import matplotlib.pyplot as plt

    # 总步数
    N = 150

    # 系统模型（1 维）
    A = 1.0  # 状态转移
    H = 1.0  # 观测矩阵
    Q = 1e-2  # 过程噪声方差（固定）

    # 观测噪声方差 R_k：分三段变化
    R_seq = np.zeros(N)
    R_seq[0:50] = 0.1  # 前 0~49 步：观测很准
    R_seq[50:100] = 4.0  # 50~99 步：观测很差
    R_seq[100:150] = 0.2  # 100~149 步：观测再好一些

    # 用来存储结果
    P_pred = np.zeros(N)  # 先验协方差
    P_upd = np.zeros(N)  # 后验协方差
    K_seq = np.zeros(N)  # 卡尔曼增益（可选）

    # 初始协方差
    P = 1.0

    for k in range(N):
        # ---------- 预测 ----------
        P = A * P * A + Q
        P_pred[k] = P

        # ---------- 更新 ----------
        Rk = R_seq[k]
        S = H * P * H + Rk
        K = P * H / S  # 卡尔曼增益
        K_seq[k] = K

        P = (1 - K * H) * P  # 约瑟夫型更新在 1D 下就是这个形式
        P_upd[k] = P

    # 画图看看协方差波动
    time = np.arange(N)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, R_seq, label="Measurement noise variance R_k")
    plt.ylabel("R_k")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, P_pred, label="P_pred (prior)")
    plt.plot(time, P_upd, label="P_upd (posterior)", linestyle="--")
    plt.xlabel("Time step k")
    plt.ylabel("Covariance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_q()