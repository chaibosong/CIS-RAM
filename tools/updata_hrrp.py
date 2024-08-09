import os
import random
import pandas as pd
import scipy.io as scio
import numpy as np


def txt2csv():
    with open('../datasets/data_all/hrrp_type_1/label.txt', 'r') as txt:
        lines = txt.readlines()
        data_txt = []
        for line in lines:
            print(line)
            img_name = line.split(' ')[0]
            frame_number = line.split(' ')[1]
            cls = line.split(' ')[2]
            x_min = line.split(' ')[3]
            y_min = line.split(' ')[4]
            x_max = line.split(' ')[5]
            y_max = line.split(' ')[6]

            # heading = random.sample(range(-180, 180), 1)
            # tilt = random.sample(range(-90, 90), 1)
            heading = round(random.uniform(-180, 180), 2)
            tilt = round(random.uniform(-90, 90), 2)
            print(heading)
            print(tilt)
            data_txt.append([img_name, frame_number, cls, heading, tilt, x_min, y_min, x_max, y_max])

    data_txtDF = pd.DataFrame(data_txt, columns=['image_name', 'frame_number', 'cls', 'heading', 'tilt', 'x_min', 'y_min', 'x_max', 'y_max'], index=None)
    data_txtDF.to_csv('../datasets/data_all/hrrp_type_1/label.csv', index=False)


def update_hrrp():
    hrrps_prv = scio.loadmat('../datasets/data_all/hrrp_type_1/hrrps_prv.mat')
    hrrps = []
    hrrp = []
    hrrps_prv_data = hrrps_prv['data']
    # for hrrp in hrrps_data:
    #     print(hrrp)
    df = pd.read_csv('../datasets/data_all/hrrp_type_1/label.csv', sep=',', usecols=[1, 2, 3, 4])
    # print(df)
    for index, row in df.iterrows():
        hrrp.append(int(row['frame_number']))
        if row['cls'] == 'car':
            hrrp.append(int(0))
            hrrp.append(int(0))
            hrrp.append(int(1))
        elif row['cls'] == 'person':
            hrrp.append(int(0))
            hrrp.append(int(1))
            hrrp.append(int(0))
        hrrp.append(row['heading'])
        hrrp.append(row['tilt'])
        # hrrp.append(hrrps_prv_data[index])
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(index)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        hrrp = hrrp + hrrps_prv_data[index].tolist()
        print(hrrp)
        print()

        hrrps.append(hrrp)
        hrrp = []
    # for hrrp in hrrps_prv_data:

    print(hrrps)
    scio.savemat('../datasets/data_all/hrrp_type_1/hrrps.mat', {'data': hrrps})




if __name__ == '__main__':
    # update_hrrp()
    # hrrps_prv = scio.loadmat('../datasets/data_all/hrrp_type_1/hrrps_prv.mat')
    #
    # print(hrrps_prv['data'][95])
    #
    # print('**********************')
    #
    hrrps = scio.loadmat('../datasets/data_all/hrrp_type_1/hrrps.mat')
    print(hrrps['data'][95])