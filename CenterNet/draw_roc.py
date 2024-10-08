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
    fpr, tpr, thread = roc_curve(y_test, y_score)

    roc_auc = auc(fpr, tpr)

    # 绘图
    plt.figure()

    plt.plot(fpr, tpr, color='red',
             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('./imgs_out/roc.png')
    # plt.close()
    plt.show()

    # y = scaler.fit_transform(data[i][64:192].reshape(-1, 1))

    # print(data)

    # x = range(0, 128)

    # plt.plot(x, y, linewidth=1, color='black')

    # plt.savefig('./imgs/pic-{}.png'.format(i + 1))
    # plt.close()
