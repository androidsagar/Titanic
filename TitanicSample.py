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

x = df[['Pclass','Sex','Age','SibSp','Parch']].values
y = df[['Survived']].values

imputer = SimpleImputer(strategy="mean")
x[:, 2:3] = imputer.fit_transform(x[:, 2:3]).astype(dtype= int)

labelEncoder_X = LabelEncoder()
x[:, 1] = labelEncoder_X.fit_transform(x[:, 2])

dfTest = pd.read_csv('/home/agl-android/.ML/Titanic/data/test.csv')
xTest =  dfTest[['Pclass','Sex','Age','SibSp','Parch']].values

imputerTest = SimpleImputer(strategy="mean")
xTest[:, 2:3] = imputerTest.fit_transform(xTest[:, 2:3]).astype(dtype= int)

labelEncoder_XTest = LabelEncoder()
xTest[:, 1] = labelEncoder_XTest.fit_transform(xTest[:, 1])

regr = LinearRegression()
regr = regr.fit(x,y)
yPridict = regr.predict(xTest)

out = pd.DataFrame()
out['PassengerId'] = dfTest['PassengerId'].values
out['Survived'] = yPridict

