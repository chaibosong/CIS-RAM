import scipy.io
import numpy as np
file_path = '/home/lll/software/fixed/MMDetector/test_data/ShangHai_4/hrrps.mat'
file_path_1 = '/home/lll/software/fixed/MMDetector/test_data/ShanghaiNewData/hrrps.mat'
data = scipy.io.loadmat(file_path)['aa']
data_1 = scipy.io.loadmat(file_path_1)['aa']
c = np.concatenate((data,data_1),axis=0)
print(data[19993])
print(data.shape)
print(c.shape)
data_2 = {
    'aa':c
}
#scipy.io.savemat('/home/lll/software/fixed/MMDetector/test_data/3_4/hrrps_1.mat',data_2)
data_3 = scipy.io.loadmat('/test_data/3_4/hrrps.mat')['aa']
print(data_3[19993])
print(data_3.shape)