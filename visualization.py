# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def vis_square(data):

    # 输入的数据为一个ndarray，尺寸可以为(n, height, width)或者是 (n, height, width, 3)
    # 前者即为n个灰度图像的数据，后者为n个rgb图像的数据
    # 在一个sqrt(n) by sqrt(n)的格子中，显示每一幅图像

    # 对输入的图像进行normlization
    data = (data - data.min()) / (data.max() - data.min())

    # 强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # 每幅小图像之间加入小空隙
    padding = (((0, n ** 2 - data.shape[0]),(0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)   

    # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # data的一个小例子,e.g., (3,120,120)
    # 即，这里的data是一个2d 或者 3d 的ndarray
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # 显示data所对应的图像
    plt.imshow(data,cmap="gray"); plt.axis('off')
    plt.savefig('C:/Users/Alienware/Desktop/visualization/IGTL.eps',dpi = 1000,bbox_inches='tight')
    plt.show()