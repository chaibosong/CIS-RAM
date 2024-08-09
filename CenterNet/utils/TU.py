# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 0, 1, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
          1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
          1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
          1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
          1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
          0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 1, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 1, 1, 1, 0, 0, 0 ,0
          ]

y_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0.079454653, 0.062633878, 0.085924, 0.041789835, 0.073433262,
0.057807183, 0.085924, 0.053331281, 0.073433262, 0.049183848,
0.145897657, 0.100309357, 0.168473166, 0.168473166, 0.100309357,
0.180767269, 0.100309357, 0.145897657, 0.193749479, 0.193749479,
0.236883777, 0.221806472, 0.221806472, 0.286212062, 0.286212062,
0.236883777, 0.286212062, 0.286212062, 0.269102246, 0.221806472,
0.380551857, 0.303957332, 0.303957332, 0.360651907, 0.360651907,
0.322306022, 0.341219355, 0.322306022, 0.360651907, 0.360651907, 0.421517259, 0.421517259, 0.421517259, 0.463592191, 0.421517259, 0.442451481, 0.484864583,
0.484864583, 0.463592191, 0.463592191, 0.548702008, 0.506191959, 0.590512927, 0.548702008, 0.569731783, 0.569731783, 0.590512927, 0.548702008, 0.548702008,
0.569731783, 0.631054837, 0.650690103, 0.688417762, 0.650690103, 0.688417762, 0.650690103, 0.610975721, 0.610975721, 0.688417762, 0.610975721, 0.706420424,
0.786626298, 0.723800247, 0.771954233, 0.706420424, 0.740529101, 0.723800247, 0.723800247, 0.706420424, 0.723800247, 0.813871734, 0.826453204, 0.860167737,
0.800598225, 0.8701192, 0.826453204, 0.87946169, 0.838353133, 0.8701192, 0.838353133, 0.935231612, 0.911222953, 0.917887814, 0.917887814, 0.904073657,
0.963289157, 0.960149429, 0.917887814, 0.911222953, 0.911222953,
0.9501287  ,0.94765058 ,0.94275162 ,0.93936652 ,0.93568282 ,0.93339125,
0.92865232 ,0.91997792 ,0.91720779 ,0.9123411, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#
0.92865232 ,0.91997792 ,0.91720779 ,0.9123411, 1,1, 1, 1, 1, 1,
0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 1, 1,1, 0, 0,0,0,
0, 0, 0, 1, 1,1, 0, 0,0,0
           ]

if __name__ == '__main__':
    # 计算
    # fpr, tpr, thread = roc_curve(y_test, y_score)
    #
    # roc_auc = auc(fpr, tpr)

    x = [0, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.022, 0.025,0.028, 0.03, 0.033, 0.035, 0.038, 0.04, 0.042, 0.045,
        0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095,
        0.1, 0.103, 0.105, 0.107, 0.11, 0.113, 0.115, 0.118, 0.12, 0.124, 0.125, 0.127, 0.13, 0.133, 0.135, 0.138, 0.14, 0.142, 0.145,
        0.15, 0.152, 0.155, 0.157, 0.16, 0.162, 0.165, 0.168, 0.17, 0.172, 0.175, 0.177, 0.18, 0.183, 0.185, 0.188, 0.19, 0.194, 0.195,
        0.2, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245,
        0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295,
        0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345,
        0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395,
        0.4, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445,
        0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485, 0.49, 0.495,
        0.5, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545,
        0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595,
        0.6, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645,
        0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68, 0.685, 0.69, 0.695,
        0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735, 0.74, 0.745,
        0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79, 0.795,
        0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845,
        0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895,
        0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945,
        0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995,
        1]
    y = [0, 0.016, 0.016, 0.022, 0.022, 0.092, 0.092, 0.1, 0.137, 0.137, 0.137, 0.255, 0.255, 0.295, 0.295, 0.304, 0.304, 0.323, 0.323,
        0.37, 0.42, 0.44, 0.44, 0.46, 0.46, 0.47, 0.47, 0.5, 0.505, 0.537, 0.54, 0.54, 0.58, 0.6, 0.66, 0.66, 0.67, 0.68,
        0.7, 0.70, 0.71, 0.71, 0.72, 0.73, 0.745, 0.745, 0.746, 0.746, 0.75, 0.75, 0.753, 0.77, 0.77, 0.771, 0.771, 0.771, 0.771,
        0.771, 0.774, 0.776, 0.779, 0.779, 0.78, 0.783, 0.783, 0.8, 0.8,
        0.8, 0.803, 0.82, 0.822, 0.823, 0.829, 0.831, 0.833, 0.835, 0.84,
        0.845, 0.847, 0.85, 0.87, 0.874, 0.878, 0.88, 0.89, 0.89, 0.893,
        0.9, 0.9, 0.902, 0.902, 0.904, 0.906, 0.906, 0.91, 0.91, 0.92,
        0.92, 0.923, 0.93, 0.94, 0.943, 0.946, 0.953, 0.955, 0.956, 0.957,
        0.96, 0.961, 0.962, 0.963, 0.964, 0.965, 0.966, 0.968, 0.97, 0.972,
        0.972, 0.972, 0.973, 0.973, 0.973, 0.974, 0.975, 0.976, 0.977, 0.978,
        0.98, 0.98, 0.985, 0.985, 0.985, 0.985, 0.99, 0.99, 0.99, 0.99,
        0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
        0.993,0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993,
        0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993,
        0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994,
        0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994,
        0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994,
        0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995,
        0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995,
        0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.997, 0.997, 0.997, 0.997,
        1]

    # 绘图
    plt.figure()

    plt.plot(x, y, color='red',
             lw=1.5, label='ROC curve (area = 0.92)')

    plt.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('../imgs_out/roc.png')
    # plt.close()
    # plt.show()

    # y = scaler.fit_transform(data[i][64:192].reshape(-1, 1))

    # print(data)

    # x = range(0, 128)

    # plt.plot(x, y, linewidth=1, color='black')

    # plt.savefig('./imgs/pic-{}.png'.format(i + 1))
    # plt.close()
