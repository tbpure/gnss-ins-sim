from time import sleep

from sim_data_gen.demo_no_algo import test_path_gen


def gen_data(file_path):
    for i in range(50):
        test_path_gen(file_path, file_path)


if __name__ == '__main__':
    gen_data('/Users/bytedance/PycharmProjects/gnss-ins-sim/cnn_gru/demo_saved_data/drone_sim')
