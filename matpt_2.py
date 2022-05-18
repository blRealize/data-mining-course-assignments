from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import true


train = pd.read_csv('train.csv')
plt.scatter(train.SalePrice, train.GarageArea)
plt.xlabel('SalePrice(k)')
plt.ylabel('GarageArea')
plt.show()