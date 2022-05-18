from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import true
from zmq import EVENT_CLOSE_FAILED


train = pd.read_csv('train.csv',usecols=['BldgType', 'SalePrice'])
# box1, box2, box3, box4, box5 = train.BldgType['1Fam'], train.BldgType['2FmCon'],
# train.BldgType['Duplx'], train.BldgType['TwnhsE'], train.Bldgtype['Twnhsl']
box1 = []
box2 = [] 
box3 = [] 
box4 = [] 
box5 = []
# row_data = train[0:1460]
for i, row in train.iterrows():
    x, y = row['X'], row['Y'] 
    print(i)
    if i['BldgType'] == '1Fam':
        box1.append(i['SalePrice'])
    elif i['BldgType'] == '2FmCon':
        box2.append(i['SalePrice'])
    elif i['BldgType'] == 'Duplx':
        box3.append(i['SalePrice'])
    elif i['BldgType'] == 'TwnhsE':
        box4.append(i['SalePrice'])
    elif i['BldgType'] == 'Twnhsl':
        box5.append(i['SalePrice'])
# box1, box2, box3, box4, box5 = train.BldgType, train.BldgType, train.BldgType, train.BldgType, train.BldgType
plt.boxplot([box1, box2, box3, box4, box5])
plt.xlabel('BldgType')
plt.ylabel('SalePrice')
plt.show()