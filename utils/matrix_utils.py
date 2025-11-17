import numpy as np


def skew(v):
    """ 反对称矩阵 """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

