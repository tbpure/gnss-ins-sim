import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def quat_normalize(q):
    return q / np.linalg.norm(q)

def small_angle_quat(omega_dt):
    """将角速度积分为小角度四元数"""
    angle = np.linalg.norm(omega_dt)
    if angle < 1e-8:
        qw = 1.0
        qv = 0.5 * omega_dt
    else:
        axis = omega_dt / angle
        qw = np.cos(angle / 2.0)
        qv = axis * np.sin(angle / 2.0)
    return quat_normalize(np.hstack((qw, qv)))

def quat_mul(q1, q2):
    # q = [w, x, y, z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def rot_from_quat(q):
    # 返回从机体到导航（NED）坐标的旋转矩阵 R_b2n
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z),    2*(x*z + w*y)],
        [2*(x*y + w*z),   1-2*(x*x+z*z),    2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x),    1-2*(x*x+y*y)]
    ])
    return R


def ins_update(state, imu_meas, dt):
    """
    纯捷联惯导积分
    state: dict, 包含
        q: 姿态四元数 (b→n)
        v: 速度 NED (m/s)
        p: 位置 NED (m)
    imu_meas: dict, 包含
        w: 角速度 (rad/s)
        a: 加速度 (m/s^2)
    dt: 时间步长 (s)
    """
    q = state['q']
    v = state['v']
    p = state['p']

    # 1. 姿态更新
    dq = small_angle_quat(imu_meas['w'] * dt)
    q = quat_mul(q, dq)
    q = quat_normalize(q)

    # 2. 旋转到导航系
    R_b2n = rot_from_quat(q)
    f_n = R_b2n.dot(imu_meas['a'])

    # 3. 加重力项（NED中重力向下为正）
    g_n = np.array([0, 0, 9.80665])
    v = v + (f_n + g_n) * dt

    # 4. 位置更新
    p = p + v * dt

    return {'q': q, 'v': v, 'p': p}


def read_data(data_path: str, ref: bool = False):
    if ref:
        acc = pd.read_csv(data_path + 'ref_accel.csv')
        gyro = pd.read_csv(data_path + 'ref_gyro.csv')
        mag = pd.read_csv(data_path + 'ref_mag.csv')
        init_quat = pd.read_csv(data_path + 'ref_att_quat.csv').iloc[0]
    else:
        acc = pd.read_csv(data_path + 'accel-0.csv')
        gyro = pd.read_csv(data_path + 'gyro-0.csv')
        mag = pd.read_csv(data_path + 'mag-0.csv')
        init_quat = pd.read_csv(data_path + 'ref_att_quat.csv').iloc[0]
    return acc.to_numpy(), gyro.to_numpy(), mag.to_numpy(), init_quat.to_numpy()


def ins(time: int):
    acc, gyro, mag, init_quat = read_data('./sim_data_gen/sim_files/saved_file/2025-10-25-22-15-07/', True)
    state = {
        'q': init_quat,
        'v': np.zeros(3),
        'p': np.zeros(3)
    }

    traj = []

    if time == -1:
        time = acc.shape[0]
    else:
        time *= 100

    for i in range(time):
        imu = {
            'a': acc[i][:],
            'w': gyro[i][:]
        }
        state = ins_update(state, imu, 0.01)
        traj.append(state['p'])
    traj = np.array(traj)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='INS Trajectory')
    ax.set_xlabel('North [m]')
    ax.set_ylabel('East [m]')
    ax.set_zlabel('Down [m]')
    ax.set_title('3D INS Trajectory')
    ax.legend()
    plt.show()

    plt.figure()
    plt.plot(traj[:, 1], traj[:, 0], label='INS Trajectory')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title('2D INS Trajectory (Top View)')
    plt.axis('equal')  # 保持比例
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ins(-1)
