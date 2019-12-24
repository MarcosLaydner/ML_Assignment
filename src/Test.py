import pandas as pd
import sklearn

data = pd.read_csv("../resources/CE802_Ass_2019_Data.csv")

features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19']

print(data.dtypes)

X = data.loc[:, features]

print(X.shape)

y = data.Class

print(y.shape)
