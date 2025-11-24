import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_io import get_imu_data_from_path, get_gnss_data_from_path

ratio = 100

def get_dataset_from_single_path(file_path, step):
    imu_data = get_imu_data_from_path(file_path, False)[ratio:][:]
    gnss_data = get_gnss_data_from_path(file_path, False)
    gnss_data = np.diff(gnss_data, axis=0)
    features = []
    labels = []
    for i in range(0, len(gnss_data) - step):
        features.append(imu_data[(i + step) * ratio - step:(i + step) * ratio][:])
        labels.append(gnss_data[i + step][:])
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