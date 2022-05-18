from dataclasses import dataclass
from re import S
from smtplib import SMTPSenderRefused
from hamcrest import same_instance
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression

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

clean_dataset(df1)
X = df1.drop(columns = ["SalePrice"])
X = X.reset_index()
#print(X)
y = df1.loc[:, 'SalePrice']
y = y.reset_index()
#print(y)

Linreg = LinearRegression()
model = Linreg.fit(X, y)
print(model)

print(Linreg.intercept_)
print(Linreg.coef_)

y_pred = Linreg.predict(X)
print(y_pred)
Smae = 0
Smse = 0
for i in range(len(y_pred)):
    Smae += abs(y_pred[i]-y.values[i])
    Smse += (y_pred[i]-y.values[i]) * (y_pred[i]-y.values[i])

mae = Smae/len(y_pred)
mse = Smse/len(y_pred)
print("MAE")
print(mae)
print("MSE")
print(mse)
# https://blog.csdn.net/HHTNAN/article/details/78843722