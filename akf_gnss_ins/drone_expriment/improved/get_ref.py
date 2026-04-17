import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = '../datas/final/ref_interp.xlsx'
pd = pd.read_excel(path)
ref = pd.to_numpy()
print(ref.shape)

north = ref[:, 0]
east = ref[:, 1]
high = ref[:-10, 2]
window_size = 8  # 可以调大一点更平滑

north = np.convolve(north, np.ones(window_size)/window_size, mode='same')[:-10]
east  = np.convolve(east,  np.ones(window_size)/window_size, mode='same')[:-10]
plt.figure(figsize=(8, 5))  # 改成长方形更适合非等比例

plt.plot(north[:-10], east[:-10], linewidth=2, label='Reference Trajectory')

# 自动根据数据范围设置边界（留一点边距更好看）
margin = 0.05

n_min, n_max = np.min(north), np.max(north)
e_min, e_max = np.min(east), np.max(east)

plt.xlim(n_min - margin*(n_max-n_min), n_max + margin*(n_max-n_min))
plt.ylim(e_min - margin*(e_max-e_min), e_max + margin*(e_max-e_min))

# 坐标轴
plt.xlabel('North (m)')
plt.ylabel('East (m)')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.title('North-East Trajectory Projection')

plt.tight_layout()
# plt.savefig("ref_x_y.png", dpi=600)
plt.show()