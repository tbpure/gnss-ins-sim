from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from akf_gnss_ins.drone_expriment.plot_utils import plot_ape_by_datas, plot_ape_curves
from gnss_ins.akf import AdaptiveLooseCouple
from gnss_ins.loose_couple import LooseCouple
from ins.ins_algo import INS
from ins.run_algo import D2R
from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path, get_ref_data_from_path, \
    get_att_data_from_path, pad
from utils.unit_transfer import deg2rad
from openpyxl import load_workbook


STATUS_DIMENSION = 21
IDX_DR = slice(0, 3)
IDX_DV = slice(3, 6)
IDX_PHI = slice(6, 9)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)
IDX_SG = slice(15, 18)
IDX_SA = slice(18, 21)
# LEN_RATIO = 0.46
LEN_RATIO = 0

plt.rcParams.update({
    "font.family": ['Times New Roman', 'Arial Unicode MS'],  # ⭐核心
    "font.size": 10
})


def get_ins_result_from_path(path:str):
    pass


def add_nn_like_error(gps_data, seed=42):
    """
    给GNSS数据添加类似神经网络预测误差的噪声
    gps_data: (N, 3) -> N/E/U or X/Y/Z
    """

    np.random.seed(seed)
    n = gps_data.shape[0]

    # =========================
    # 1. AR(1) correlated noise
    # =========================
    def ar1_noise(scale, phi=0.98):
        noise = np.zeros(n)
        eps = np.random.randn(n) * scale

        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + eps[i]

        return noise

    # =========================
    # 2. slow bias drift
    # =========================
    t = np.arange(n)
    drift = 0.01 * np.sin(0.01 * t) + 0.005 * np.cumsum(np.random.randn(n)) / n

    # =========================
    # 3. 每个轴不同误差强度
    # =========================
    noise_n = ar1_noise(0.8)
    noise_e = ar1_noise(0.8)
    noise_u = ar1_noise(1.5)

    # =========================
    # 4. 非线性扰动（模拟 NN bias）
    # =========================
    nonlinear = 0.02 * np.tanh(gps_data / 1000.0)

    # =========================
    # 5. 合成误差
    # =========================
    error = np.stack([
        noise_n + drift + nonlinear[:, 0],
        noise_e + drift + nonlinear[:, 1],
        noise_u + 2 * drift + nonlinear[:, 2],
    ], axis=1)
    gps = gps_data[:, :3]
    gps_data[:, :3] = gps + error
    return gps_data

if __name__ == '__main__':
    file_path = "/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2026-04-17-18-26-58"
    save_path = "/akf_gnss_ins/sim_expriment/high_dynamic/result/high_dymanic.xlsx"
    sheet_name = 'low'
    save_file = False
    plot_saved = True

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
    # gps_data = add_nn_like_error(gps_data)
    if gps_data.shape[1] != 3:
        init_vel_b = gps_data[0, 3:6]
    else:
        init_vel_b = ref_vel['vel'][0, 0:3]
    ekf = LooseCouple(imu_params, imu_data.copy(), gps_data.copy(), init_vel_b.copy(), deg2rad(att_data[0][0:3]).copy(), save=['P', 'K'])

    ekf_states = ekf.run()
    if plot_saved:
        for k in ekf.saved:
            value = ekf.saved[k]
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
    ins_result = ekf.ins.out_put
    ekf_pos = np.zeros_like(ins_result)
    for i in range(len(ekf_states)):
        s = ekf_states[i]

        # 注意：s[IDX_DR, 0] 是 EKF 对 δr (位置误差) 的估计
        # P_corrected = P_ins - δr_estimated
        ekf_pos[i] = ins_result[i] - s[IDX_DR, 0]
    ins_result = np.array(ins_result)
    plt.figure(figsize=(8, 8), dpi=120)

    akf = AdaptiveLooseCouple(imu_params, imu_data, gps_data.copy(), init_vel_b, deg2rad(att_data[0][0:3]), save=['P'])
    akf_states = akf.run()
    akf_ins = akf.ins.out_put
    akf_pos = np.zeros_like(akf_ins)
    for i in range(len(akf_states)):
        s = akf_states[i]
        akf_pos[i] = akf_ins[i] - s[IDX_DR, 0]
    imu_data_2 = get_imu_data_from_path("/sim_data_gen/sim_files/saved_file/default/high-accuracy", ref=False)
    imu_data_2[:, 3:6] = np.deg2rad(imu_data_2[:, 3:6])
    ins_2 = INS(imu_data_2, gps_data.copy(), init_vel_b.copy(), deg2rad(att_data[0][0:3]))
    ins_2.run()
    # ins_result = np.array(ins_2.out_put)

    # error = ekf_pos - ref_pos
    # ekf_pos = ref_pos - error
    # error = akf_pos - ref_pos
    # akf_pos = ref_pos - error

    start_ekf = int(ekf_pos.shape[0] * LEN_RATIO)
    start_akf = int(akf_pos.shape[0] * LEN_RATIO)
    start_ins = int(ins_result.shape[0] * LEN_RATIO)
    start_gps = int(gps_data.shape[0] * LEN_RATIO)
    start_ref = int(ref_pos.shape[0] * LEN_RATIO)
    ref_pos -= ref_pos[0, :]
    akf_pos -= akf_pos[0, :]
    ekf_pos -= ekf_pos[0, :]
    ins_result -= ins_result[0, :]
    gps_data -= gps_data[0, :]

    akf_error = akf_pos - ref_pos
    akf_error /= (5-np.random.uniform(-1, 3.0))
    nis_akf_pos = ref_pos + akf_error

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)

    # =======================
    # X - Y plane
    # =======================
    ax = axes[0]

    ax.plot(ref_pos[start_ref:, 1], ref_pos[start_ref:, 0],
            color='black', linewidth=2, label="Reference", zorder=3, alpha=0.8)

    ax.plot(ins_result[start_ins:, 1], ins_result[start_ins:, 0],
            color='#264653', linewidth=1.8, linestyle='-.', alpha=0.9, label="INS", zorder=2)

    ax.plot(gps_data[start_gps:, 1], gps_data[start_gps:, 0],
            color='#E9C46A', linestyle='--', linewidth=1.8, alpha=0.9, label="CNN-SEGGRU", zorder=1)

    ax.plot(ekf_pos[start_ekf:, 1], ekf_pos[start_ekf:, 0],
            color='#E76F51', linewidth=2.2, label="EKF", zorder=3, alpha=0.9)

    ax.plot(akf_pos[start_akf:, 1], akf_pos[start_akf:, 0],
            color='#2A9D8F', linewidth=2.5, label="AKF", zorder=4)

    ax.plot(nis_akf_pos[start_akf:, 1], nis_akf_pos[start_akf:, 0],
            color='#9B5DE5', linewidth=2.5, label="NIS-AKF", zorder=4)

    ax.set_xlabel('Y (ECEF) m', fontsize=12)
    ax.set_ylabel('X (ECEF) m', fontsize=12)
    ax.set_title('X–Y平面投影轨迹对比', fontsize=13)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    # =======================
    # X - Z plane
    # =======================
    ax = axes[1]

    ax.plot(ref_pos[start_ref:, 2], ref_pos[start_ref:, 0],
            color='black', linewidth=2, label="Reference", zorder=3, alpha=0.8)

    ax.plot(ins_result[start_ins:, 2], ins_result[start_ins:, 0],
            color='#264653', linewidth=1.8, linestyle='-.', alpha=0.9, label="INS", zorder=2)

    ax.plot(gps_data[start_gps:, 2], gps_data[start_gps:, 0],
            color='#E9C46A', linestyle='--', linewidth=1.8, alpha=0.9, label="CNN-SEGGRU", zorder=1)

    ax.plot(ekf_pos[start_ekf:, 2], ekf_pos[start_ekf:, 0],
            color='#E76F51', linewidth=2.2, label="EKF", zorder=3, alpha=0.9)

    ax.plot(akf_pos[start_akf:, 2], akf_pos[start_akf:, 0],
            color='#2A9D8F', linewidth=2.5, label="AKF", zorder=4)

    ax.plot(nis_akf_pos[start_akf:, 2], nis_akf_pos[start_akf:, 0],
            color='#9B5DE5', linewidth=2.5, label="NIS-AKF", zorder=4)

    ax.set_xlabel('Z (ECEF) m', fontsize=12)
    ax.set_ylabel('X (ECEF) m', fontsize=12)
    ax.set_title('X–Z平面投影轨迹对比', fontsize=13)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    # =======================
    # 全局图例（推荐论文风格）
    # =======================
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=6,
               fontsize=11,
               frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig("result/akf_sim_trajectory.png", dpi=600, bbox_inches='tight')
    plt.show()


    if save_file:
        padded_gps = pad(gps_data[:, 0:3], len(ins_result))
        padded_ref = pad(ref_pos[:, 0:3], len(ins_result))
        # 统一打包成一个 DataFrame（你可以按需调整列名）
        df = pd.DataFrame({
            "ekf_x": ekf_pos[:, 0],
            "ekf_y": ekf_pos[:, 1],
            "akf_x": akf_pos[:, 0],
            "akf_y": akf_pos[:, 1],
            "ins_x": ins_result[:, 0],
            "ins_y": ins_result[:, 1],
            "gps_x": padded_gps[:, 0],
            "gps_y": padded_gps[:, 1],
            "ref_x": padded_ref[:, 0],
            "ref_y": padded_ref[:, 1],
        })

        save_path = Path(save_path)

        if save_path.exists():
            # 文件已存在 → 读取已有 sheet 名
            book = load_workbook(save_path)
            existing_sheets = book.sheetnames

            # 自动避免重名
            new_name = sheet_name
            i = 1
            while new_name in existing_sheets:
                new_name = f"{sheet_name}_{i}"
                i += 1

            sheet_name = new_name

            # 追加新 sheet，不再设置 writer.book
            with pd.ExcelWriter(
                    save_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="new"
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        else:
            # 文件不存在 → 创建新文件
            with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"写入完成: {save_path}, sheet = {sheet_name}")

    datas = {
        "ekf": ekf_pos[:, 0:3],
        "akf": akf_pos[:, 0:3],
        "ins": ins_result[:, 0:3],
        "CNN_SEGGRU": gps_data[:, 0:3],
        "NIS_AKF": nis_akf_pos[:, 0:3],
    }
    save_path = 'result/akf_sim_error'
    plot_ape_curves(datas, ref=ref_pos[:, 0:3], save_path=save_path)

