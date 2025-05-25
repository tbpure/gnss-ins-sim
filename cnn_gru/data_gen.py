from time import sleep

from sim_data_gen.demo_no_algo import test_path_gen


def gen_data():
    for i in range(50):
        sleep(1.5)
        test_path_gen()


if __name__ == '__main__':
    gen_data()
