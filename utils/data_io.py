from pathlib import Path

import numpy as np
import pandas as pd


def get_data_from_path(path, file_list:list[str], ref:bool = False):
    path = Path(path)
    file_dict = {}
    file_name_to_type = {}
    file_name_list = []
    if not ref:
        for file in file_list:
            file_name = file + '-0.csv'
            file_name_list.append(file_name)
            file_name_to_type[file_name] = file
    else:
        for file in file_list:
            if file == 'att':
                file_name_euler = 'ref_att_euler.csv'
                file_name_to_type[file_name_euler] = 'att_euler'
                file_name_quat = 'ref_att_quat.csv'
                file_name_to_type[file_name_quat] = 'att_quat'
                file_name_list.append(file_name_euler)
                file_name_list.append(file_name_quat)
                continue
            file_name = 'ref_' + file + '.csv'
            file_name_list.append(file_name)
            file_name_to_type[file_name] = file

    for file_name in file_name_list:
        file_path = path.joinpath(file_name)
        df = pd.read_csv(file_path)
        file_dict[file_name_to_type[file_name]] = df.to_numpy()
    return file_dict


def get_imu_data_from_path(path, ref:bool = False):
    file_list = ['accel', 'gyro', 'mag']
    data_dict = get_data_from_path(path, file_list, ref = ref)
    return concat_data_from_dict(data_dict)

def get_gnss_data_from_path(path, ref:bool = False):
    file_list = ['gps']
    if ref == True:
        file_list = ['pos']
    data_dict = get_data_from_path(path, file_list, ref = ref)
    return concat_data_from_dict(data_dict)

def get_att_data_from_path(path, ref:bool = False):
    file_list = ['att']
    data_dict = get_data_from_path(path, file_list, ref = True)
    return concat_data_from_dict(data_dict)


def concat_data_from_dict(data_dict: dict[str, np.ndarray]):
    arrays = list(data_dict.values())  # 提取所有数组
    # 检查行数是否一致
    n_rows = arrays[0].shape[0]
    if all(arr.shape[0] == n_rows for arr in arrays):
        return np.hstack(arrays)
    else:
        raise ValueError("所有数组的行数必须相同才能按列拼接")


def get_ref_data_from_path(path, file_list:list[str]):
    return get_data_from_path(path, file_list, ref = True)

if __name__ == '__main__':
    imu_data = get_imu_data_from_path('/Users/yangyu/PycharmProjects/gnss-ins-sim/cnn_gru/demo_saved_data/drone_sim/2025-06-03-09-45-29')