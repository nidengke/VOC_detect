#coding:utf-8
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
N = 50  # 点的个数
x = np.random.rand(N) * 2  # 随机产生50个0~2之间的x坐标
y = np.random.rand(N) * 2  # 随机产生50个0~2之间的y坐标
colors = np.random.rand(N)  # 随机产生50个0~1之间的颜色值
area = np.pi * (15 * np.random.rand(N)) ** 2  # 点的半径范围:0~15
# 画散点图
plt.scatter(x, y, s=area, c=colors, alpha=0.5, marker=(9, 3, 30))
plt.show()