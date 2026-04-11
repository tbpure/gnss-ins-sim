import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def load_and_smooth(file_path, smooth_factor=0.1):
    df = pd.read_csv(file_path, header=None)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # 排序
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    # 去重
    x_unique, indices = np.unique(x, return_index=True)
    y_unique = y[indices]

    # 统一时间轴
    x_new = np.arange(0, 71, 1)

    # 平滑样条
    s = len(x_unique) * np.var(y_unique) * smooth_factor
    spline = UnivariateSpline(x_unique, y_unique, s=s)
    y_new = spline(x_new)

    return x_new, y_new


def load_and_average(file_path, window=0.5):
    df = pd.read_csv(file_path, header=None)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # 排序
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    # 目标 x（0~70）
    x_new = np.arange(0, 71, 1)
    y_new = []

    for xi in x_new:
        # 在窗口范围内找点
        mask = (x >= xi - window) & (x < xi + window)
        y_subset = y[mask]

        if len(y_subset) > 0:
            y_new.append(np.mean(y_subset))
        else:
            y_new.append(np.nan)

    y_new = np.array(y_new)

    # 插值补全 NaN（避免断点）
    nan_mask = np.isnan(y_new)
    y_new[nan_mask] = np.interp(x_new[nan_mask], x_new[~nan_mask], y_new[~nan_mask])

    return x_new, y_new