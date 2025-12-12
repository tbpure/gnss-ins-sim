import logging
import os.path
from configparser import MAX_INTERPOLATION_DEPTH

import numpy as np
import torch

from sim_data_gen.gnss_ins_sim.geoparams.geoparams import lla2ecef_batch
from utils import list_immediate_subdirectories
import pandas as pd


def get_imu_data(sim_only:bool=True, file_path:str=''):

    if file_path == '':
        dirs = list_immediate_subdirectories('./demo_saved_data')
    else:
        dirs = list_immediate_subdirectories(file_path)
    if not sim_only:
        file_list = ["ref_accel.csv", "ref_gyro.csv", "ref_mag.csv"]
    else:
        file_list = ["accel-0.csv", "gyro-0.csv", "mag-0.csv"]
    all_tensors = {file: [] for file in file_list}
    _len = 0
    res_tensor = torch.tensor([])
    for dir in dirs:
        for file in file_list:
            file_path = dir + "/" + file
            if not os.path.exists(file_path):
                print(f"file not exists{file_path}")
            df = pd.read_csv(file_path)
            if _len == 0:
                # 主要是为了做imu、GNSS频率比例上的统一
                # 实验比例为1:100，截断最后1s内采样时间不足1s的数据
                _len = int(df.shape[0] / 100)

            data_array = df.to_numpy().astype(np.float32)
            tensor = torch.from_numpy(data_array)
            # todo 暂时粗暴处理长度
            # tensor = tensor[:11000]
            all_tensors[file].append(tensor)
    for file, tensors in all_tensors.items():
        if tensors:
            stacked_tensor = torch.stack(tensors, dim=0)
            res_tensor = torch.cat((res_tensor, stacked_tensor), dim=2)
    logging.info("dkfjalkda")
    return res_tensor[:, :_len * 100, :]


def get_gnss_data(sim_only:bool=True, file_path:str=''):
    if file_path == '':
        dirs = list_immediate_subdirectories('./demo_saved_data')
    else:
        dirs = list_immediate_subdirectories(file_path)
    if not sim_only:
        gps_file = "ref_gps.csv"
    else:
        gps_file = "gps-0.csv"
    res_tensor = torch.tensor([])
    for dir in dirs:
        file_path = dir + "/" + gps_file
        resize = False
        if not os.path.exists(file_path):
            file_path = dir + "/" + 'ref_pos.csv'
            resize = True
        df = pd.read_csv(file_path)
        data_array = df.to_numpy().astype(np.float32)
        tensor = torch.from_numpy(data_array)
        tensor = tensor.reshape((1, tensor.shape[0], tensor.shape[1]))
        # todo 暂时粗暴处理长度
        # tensor = tensor[:, :110, :]
        if resize:
            tensor = tensor[:, ::100, :]
        res_tensor = torch.cat((res_tensor, tensor), dim=0)
    return res_tensor[:, :res_tensor.shape[1] - 1, :3]


def cal_gnss_increment_with_batch(gnss_data):
    len = gnss_data.shape[1]
    for i in range(len - 1):
        gnss_data[:, i, :] = gnss_data[:, i + 1, :] - gnss_data[:, i, :]
    return gnss_data[:, :len - 1, :]


def get_input_output_data(pre_len: int, file_path:str='') -> tuple[torch.Tensor, torch.Tensor]:
    imu_data = get_imu_data(sim_only=False, file_path = file_path)
    gnss_data = get_gnss_data(sim_only=False, file_path=file_path)
    gnss_incre = cal_gnss_increment_with_batch(gnss_data)
    assert gnss_incre.shape[1] == gnss_data.shape[1] - 1
    imu_data = imu_data[:, :gnss_incre.shape[1] * 100, :]
    if pre_len > 100:
        raise Exception("pre_len must be less than 100")
    input_slices = []
    for i in range(gnss_incre.shape[1]):
        start_idx = i * 100
        end_idx = start_idx + pre_len
        # 提取 imu 数据片段，并添加一个新维度作为 channel 维度
        slice_ = imu_data[:, start_idx:end_idx, :]  # shape: (B, pre_len, C)
        slice_ = slice_.unsqueeze(3)  # shape: (B, pre_len, C, 1)
        input_slices.append(slice_)
    input_data = torch.cat(input_slices, dim=3)
    return input_data, gnss_incre


if __name__ == '__main__':
    get_input_output_data(100, '/Users/bytedance/PycharmProjects/gnss-ins-sim/cnn_gru/demo_saved_data/sim_2')
