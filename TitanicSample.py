#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:32:30 2019

@author: agl-android
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/agl-android/.ML/Titanic/data/train.csv')

x = df[['PassengerId','Pclass','Sex','Age','SibSp','Parch']].values
y = df[['Survived']].values

imputer = SimpleImputer(strategy="mean")
x[:, 3:4] = imputer.fit_transform(x[:, 3:4]).astype(dtype= int)

labelEncoder_X = LabelEncoder()
x[:, 2] = labelEncoder_X.fit_transform(x[:, 2])

dfTest = pd.read_csv('/home/agl-android/.ML/Titanic/data/test.csv')
xTest =  dfTest[['PassengerId','Pclass','Sex','Age','SibSp','Parch']].values

imputerTest = SimpleImputer(strategy="mean")
xTest[:, 3:4] = imputerTest.fit_transform(xTest[:, 3:4]).astype(dtype= int)

labelEncoder_XTest = LabelEncoder()
xTest[:, 2] = labelEncoder_XTest.fit_transform(xTest[:, 2])

regr = LinearRegression()
regr = regr.fit(x,y)
yPridict = regr.predict(xTest).astype(dtype = int)

out = pd.DataFrame()
out['PassengerId'] = xTest[:,0]
out['Survived'] = yPridict

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(yPridict, y)
