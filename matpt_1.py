from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import true


train = pd.read_csv('train.csv')
train.SalePrice = train.SalePrice / 1000
bins = [0,100,200,300,400,500,600,700,800]
plt.hist(train.SalePrice, bins = bins, rwidth = .7)
plt.xlabel('SalePrice(k)')
plt.ylabel('#Records')
plt.show()