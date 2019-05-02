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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/agl-android/.ML/Titanic/data/train.csv')

x = df[['Pclass','Sex','Age','SibSp','Parch','Embarked']].values
y = df[['Survived']].values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.30)

imputer = SimpleImputer(strategy="mean")
xTrain[:, 2:3] = imputer.fit_transform(xTrain[:, 2:3]).astype(dtype= int)

imputer = SimpleImputer(strategy="most_frequent")
xTrain[:, 5:6] = imputer.fit_transform(xTrain[:, 5:6])


labelEncoder_X = LabelEncoder()
xTrain[:, 1] = labelEncoder_X.fit_transform(xTrain[:, 1])


dum = pd.get_dummies(xTrain[:,5],drop_first = True)

xTrain = xTrain[:,:-1]

xTrain = np.append(xTrain,dum, axis=1)

# process Test Data

imputerTest = SimpleImputer(strategy="mean")
xTest[:, 2:3] = imputerTest.fit_transform(xTest[:, 2:3]).astype(dtype= int)

imputerTest = SimpleImputer(strategy="most_frequent")
xTest[:, 5:6] = imputerTest.fit_transform(xTest[:, 5:6])

labelEncoder_XTest = LabelEncoder()
xTest[:, 1] = labelEncoder_XTest.fit_transform(xTest[:, 1])

dumTest = pd.get_dummies(xTest[:,5],drop_first = True)

xTest = xTest[:,:-1]

xTest = np.append(xTest,dumTest, axis=1)

#Scale 

scaler = StandardScaler()
xTrainFinal = scaler.fit_transform(xTrain)
xTestFinal = scaler.transform(xTest) 


import statsmodels.api as sm
xTrainSm = xTrainFinal
xTrainSm = sm.add_constant(xTrainSm)
x_opt = xTrainSm[:,[0,1,2,3,4]]
regr_OLS = sm.OLS(endog = yTrain,exog = x_opt).fit() 
regr_OLS.summary()


regr = LinearRegression()
regr.fit(xTrainFinal[:,[0,1,2,3,4]],yTrainFinal)
yPredict = regr.predict(xTestFinal[:,[0,1,2,3,4]])

r_square = r2_score(yTestFinal, yPredict)



 

#
#dfTest = pd.read_csv('/home/agl-android/.ML/Titanic/data/test.csv')
#xTest =  dfTest[['Pclass','Sex','Age','SibSp','Parch','Embarked']].values
#



#scaler = StandardScaler()
#xTrainFinal = scaler.fit_transform(x)
#xTestFinal = scaler.transform(xTest) 




out = pd.DataFrame()
out['PassengerId'] = dfTest['PassengerId'].values
out['Survived'] = yPridict

