from dataclasses import dataclass
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from re import S
from smtplib import SMTPSenderRefused
from hamcrest import same_instance
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sympy import linsolve
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# train_index = np.random.choice(150, 120, replace = False)
# validate_index = np.array(list(set(range(150)) - set(train_index))) 

df1 = pd.read_csv('train.csv', encoding='utf-8')
# print(df1.dtypes)
index = []
for i in df1.columns.values:
    if df1[i].dtype != 'object':
        index.append(i)
# print(index)
df1 = df1.loc[:, index]
df1 = df1.sample(frac=1.0)  # 全部打乱
cut_index = int(round(0.2 * df1.shape[0]))
df1_valid, df1_train = df1.iloc[:cut_index], df1.iloc[cut_index:]
df1_valid.to_csv("./validate_num.csv", index=False)
df1_train.to_csv("./train_num.csv", index=False)
# https://blog.51cto.com/u_15426866/4567811
# https://www.bbsmax.com/A/obzb0ANj5E/

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
    # https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for

clean_dataset(df1_train)
X = df1_train.drop(columns = ["SalePrice"])
X = X.reset_index()
#print(X)
y = df1_train.loc[:, 'SalePrice']
y = y.reset_index()
#print(y)

Linreg = LinearRegression()
model = Linreg.fit(X, y)
# print(model)

# print(Linreg.intercept_)
# print(Linreg.coef_)
clean_dataset(df1_valid)
X_valid = df1_valid.drop(columns = ["SalePrice"])
X_valid = X_valid.reset_index()
y_valid = df1_valid.loc[:, 'SalePrice']
y_valid = y_valid.reset_index()
y_pred = Linreg.predict(X_valid)

# print(y_pred)
Smae = 0
Smse = 0
for i in range(len(y_pred)):
    Smae += abs(y_pred[i]-y_valid.values[i])
    Smse += (y_pred[i]-y_valid.values[i]) * (y_pred[i]-y_valid.values[i])

mae = Smae/len(y_pred)
mse = Smse/len(y_pred)
print("MAE")
print(mae)
print("MSE")
print(mse)

df2_train = df1_train
df2_valid = df1_valid
# for i, row in df2_train.iterrows():
    # tempx = row['SalePrice']
df2_train['SalePrice'] = ((df2_train['SalePrice'] + 99999) / 100000)
df2_train['SalePrice'] = np.floor(pd.to_numeric(df2_train['SalePrice'], errors='coerce')).astype('Int64')

df2_valid['SalePrice'] = ((df2_valid['SalePrice'] + 99999) / 100000)
df2_valid['SalePrice'] = np.floor(pd.to_numeric(df2_valid['SalePrice'], errors='coerce')).astype('Int64')

clean_dataset(df2_train)
clean_dataset(df2_valid)

X2_train = df2_train.drop(columns = ["SalePrice"])
X2_train = X2_train.reset_index()
# y2_train = df2_train.loc[:, 'SalePrice']
y2_train = pd.DataFrame(data = df2_train, columns=['SalePrice'])
y2_train_list = y2_train.values.ravel()
# y2_train = y2_train.reset_index()
print(X2_train)
print(y2_train)
# ----------------------------------
# SVM

model_svm=svm.SVC(kernel = 'linear') 
# print("ok0\n")
model_svm.fit(X2_train, y2_train_list.astype('int'))  

# print("ok1\n")
X2_valid = df2_valid.drop(columns = ["SalePrice"])
X2_valid = X2_valid.reset_index()
y2_valid = pd.DataFrame(data = df2_valid, columns=['SalePrice'])
# print("ok2\n")
y2_pred = model_svm.predict(X2_valid)
# print("ok3\n")
print(y2_pred)

correctnum = 0
for i in range(len(y2_pred)):
    if y2_pred[i] == y2_valid.values[i]:
        correctnum = correctnum + 1
print('SVM:')
print(correctnum / len(y2_pred))

# -----------------------------------
# logistic regression
model_log = LogisticRegression()
model_log.fit(X2_train, y2_train_list.astype('int'))
y3_pred = model_log.predict(X2_valid)
print(y3_pred)

correctnum_log = 0
for i in range(len(y3_pred)):
    if y3_pred[i] == y2_valid.values[i]:
        correctnum_log = correctnum_log + 1
print('LogisticRegression:')
print(correctnum_log / len(y3_pred))
# https://www.cnpython.com/qa/124996

# https://blog.csdn.net/HHTNAN/article/details/78843722