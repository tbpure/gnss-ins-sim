import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path

ratio = 100

def get_dataset_from_single_path(file_path, step):
    # step指gnss频率，这里是1s
    if step < 1:
        raise ValueError("step should be a positive integer")
    imu_data = get_imu_data_from_path(file_path, False)[ratio:][:]
    gnss_data = get_gnss_data_from_path(file_path, False)
    gnss_data = np.diff(gnss_data, axis=0)
    features = []
    labels = []
    for i in range(step - 1, len(gnss_data) - 1):
        labels.append(gnss_data[i, :])
        features.append(imu_data[(i - step + 1)*ratio:(i + 1)*ratio, :])

    # 加入归一化
    N, T, F = features.shape

    # ========== 1. 特征归一化 ==========
    scaler_x = StandardScaler()
    features_2d = features.reshape(N * T, F)   # (N*T, F)
    features_norm_2d = scaler_x.fit_transform(features_2d)
    features_norm = features_norm_2d.reshape(N, T, F)

    # ========== 2. 标签归一化 ==========
    scaler_y = StandardScaler()
    labels_norm = scaler_y.fit_transform(labels)

    features = features_norm.copy()
    labels = labels_norm.copy()


    return np.array(features, dtype=np.float32), \
           np.array(labels, dtype=np.float32)


def load_dataset_from_paths(path_list, step):
    all_features = []
    all_labels = []

    for file_path in path_list:
        f, l = get_dataset_from_single_path(file_path, step)
        all_features.append(f)
        all_labels.append(l)

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels, axis=0)

    return features, labels


import os


def get_subfolders(parent_dir):
    """
    获取指定文件夹下的所有子文件夹（不递归）

    :param parent_dir: 父文件夹路径
    :return: 子文件夹路径列表
    """
    subfolders = [os.path.join(parent_dir, f)
                  for f in os.listdir(parent_dir)
                  if os.path.isdir(os.path.join(parent_dir, f))]
    return subfolders

class GNSSIMUDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)  # (N, step, imu_dim)
        self.labels = torch.from_numpy(labels)      # (N, gnss_dim)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



def build_dataloader(paths, step, batch_size=64, shuffle=True):
    features, labels = load_dataset_from_paths(paths, step)
    dataset = GNSSIMUDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader