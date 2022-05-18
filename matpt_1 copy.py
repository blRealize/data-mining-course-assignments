from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import true

# x = [1,2,3]
# y = [2,4,6]
# plt.plot(x,y)
# plt.show()
train = pd.read_csv('train.csv')
train.SalePrice = train.SalePrice / 1000
# bins = [0,100000,200000,300000,400000,500000,600000,700000,800000]
xrg = ['[0,100)','[100,200)','[200,300)','[300,400)','[400,500)','[500,600)','[600,700)','[700,800]']
# bins = 'auto'
# x = range(len(xrg))
plt.hist(train.SalePrice, bins = len(xrg), rwidth = .7)
# plt.xticks(train.SalePrice, xrg, rotation = 45)
plt.xlabel('SalePrice(k)')
plt.ylabel('#Records')
plt.show()

# 那个横坐标上面的坐标暂时有点处理不来，先放一下吧