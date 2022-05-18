import numpy as np
import csv

train_index = np.random.choice(150, 120, replace = False)
validate_index = np.array(list(set(range(150)) - set(train_index))) 
# 这里list的无序性是由哪个属性保证的啊
print(train_index)
print(validate_index)

with open("train.csv") as ori_data:
    ori_reader = csv.reader("train.csv")
    