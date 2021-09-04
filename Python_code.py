# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 22:17:38 2021

@author: Professor
"""

import pandas as pd
import pickle
data = pd.read_csv('CE_train.csv')
data= data.drop(['Unnamed: 0'], axis=1)
data = pd.get_dummies(data, columns = ['country'],drop_first=True)
data.drop("timestamp",axis=1, inplace=True)
data['sector'].replace(['Power','Industry','Ground Transport','Residential','International Aviation','Domestic Aviation'],[6,5,4,3,2,1],inplace= True)
y = data.value
x = data.drop('value', axis =1)
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(x,y)
pickle.dump(model,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))