from gnss_ins.gpt_demo import gpt_test_data

gnss_data, ekf_input_stream, gnss_sparse = gpt_test_data()
acc = []
gyro = []
cnb = []
vel_ins = []
pos_ins = []
for ins_data in ekf_input_stream:
    acc.append(ins_data['acc_measured'])
    gyro.append(ins_data['gyro_measured'])
    cnb.append(ins_data['Cnb_ins'])
    vel_ins.append(ins_data['vel_ins'])
    pos_ins.append(ins_data['pos_ins'])
pass