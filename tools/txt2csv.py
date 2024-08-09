import numpy as np
import pandas as pd


'''
    将txt文件转为csv
'''
# data_txt = np.loadtxt('D:/Projects/Python/MultiModel/datasets/data_all/hrrp_type_2/label.txt')


# data_txtDF = pd.DataFrame(data_txt)

with open('D:/Projects/Python/datasets/data_all/hrrp/label.txt', 'r') as txt:
    lines = txt.readlines()

    data_txt = []

    for line in lines:
        img = line.split(' ')[0]
        hrrp = line.split(' ')[1]
        cls = line.split(' ')[2]
        x_min = line.split(' ')[3]
        y_min = line.split(' ')[4]
        x_max = line.split(' ')[5]
        y_max = line.split(' ')[6]
        # data_txt.append(line.split(' '))
        data_txt.append([img, hrrp, cls, x_min, y_min, x_max, y_max])

data_txtDF = pd.DataFrame(data_txt, columns=['image', 'hrrp', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'], index=None)
data_txtDF.to_csv('D:/Projects/Python/datasets/data_all/hrrp/label.csv', index=False)
