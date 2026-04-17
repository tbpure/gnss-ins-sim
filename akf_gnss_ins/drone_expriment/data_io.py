import os

import pandas as pd

from sim_data_gen.gnss_ins_sim.geoparams.geoparams import lla2ecef_batch
from sim_data_gen.gnss_ins_sim.sim.sim_data_plot import plot_array
from utils.data_io import get_data_from_path
from utils.plot_utils import plot_3d_trajectory

file_path = "datas/origin_position.csv"

df = pd.read_csv(file_path)
array = df.to_numpy()
ecef = lla2ecef_batch(array)
ecef[:, 0] -= ecef[0, 0]
ecef[:, 1] -= ecef[0, 1]
ecef[:, 2] = array[:, 2]
# ecef /= 60
filter_data = ecef[23:-4, :]

# 4. 保存filter_data到CSV文件
# 构造DataFrame并添加列名，方便后续查看
filter_df = pd.DataFrame(
    filter_data,
    columns=["ECEF_X_offset", "ECEF_Y_offset", "Height_original"]
)
save_file_name = "datas/filter_ecef_data.csv"
current_dir = os.getcwd()
# 拼接完整路径
full_save_path = os.path.join(current_dir, save_file_name)
# 保存文件
filter_df.to_csv(full_save_path, index=False, encoding="utf-8")

plot_3d_trajectory(filter_data)
print(filter_data.shape)
print(1)