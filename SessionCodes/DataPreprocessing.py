import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df1 = pd.read_csv("Data.csv")

imputer = SimpleImputer(strategy = 'mean')
out = imputer.fit_transform(df1[['Salary', 'Age']])

df1['Salary'] = out[:,0]
df1['Age'] = out[:,1]

df1['Age'] = round(df1['Age'], 0)
df1['Salary'] = round(df1['Salary'], 0)

lenc = LabelEncoder()
df1['Purchased'] = lenc.fit_transform(df1['Purchased'])
oenc = OneHotEncoder()
enc_country = oenc.fit_transform(df1[['Country']]).toarray

sal_name = ['low', 'mid', 'high']
bins = np.linspace(df1.Salary.min(), df1.Salary.max(), 4)
df1['Sal_grp'] = pd.cut(df1.Salary, bins, labels = sal_name, include_lowest = True)
x1 = df1[['Age', 'Salary']]
y1 = df1['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)