from enum import auto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import true
from yaml import BlockSequenceStartToken
from zmq import EVENT_CLOSE_FAILED


train = pd.read_csv('train.csv',usecols=['BldgType', 'SalePrice'])
# box1, box2, box3, box4, box5 = train.BldgType['1Fam'], train.BldgType['2FmCon'],
# train.BldgType['Duplx'], train.BldgType['TwnhsE'], train.Bldgtype['Twnhsl']
box1 = []
box2 = [] 
box3 = [] 
box4 = [] 
box5 = []
for i, row in train.iterrows():
    x, y = row['BldgType'], row['SalePrice'] 
    if x == 'Twnhs':
        box1.append(y)
    elif x == '2fmCon':
        box2.append(y)
    elif x == 'TwnhsE':
        box3.append(y)
    elif x == '1Fam':
        box4.append(y)
    elif x == 'Duplex':
        box5.append(y)
# box1, box2, box3, box4, box5 = train.BldgType, train.BldgType, train.BldgType, train.BldgType, train.BldgType
x = ['','Twnhs','2fmCon','TwnhsE','1Fam','Duplex'] 
plt.boxplot([box1, box2, box3, box4, box5], boxprops = {'color':'blue'},medianprops = {'color':'green'}  )
plt.xticks(range(6),x)
plt.xlabel('BldgType')
plt.ylabel('SalePrice')
plt.show()