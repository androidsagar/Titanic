#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:32:30 2019

@author: agl-android
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
from sklearn.impute import SimpleImputer
from sklearn import tree


#df = pd.read_csv('/home/agl-android/.ML/Titanic/data/train.csv')
df = pd.read_csv('D:/MachineLearning/Project/Titanic/data/train.csv')
dfOut = pd.read_csv('D:/MachineLearning/Project/Titanic/data/gender_submission.csv') 

x = df[['Pclass','Sex','Age','SibSp','Parch','Embarked']].values
y = df[['Survived']].values

imputer = SimpleImputer(strategy="mean")
x[:, 2:3] = imputer.fit_transform(x[:, 2:3]).astype(dtype= int)

imputer1 = SimpleImputer(strategy="most_frequent")
x[:, 5:6] = imputer1.fit_transform(x[:, 5:6])


labelEncoder_Sex = LabelEncoder()
x[:, 1] = labelEncoder_Sex.fit_transform(x[:, 1])

labelEncoder_Embarked = LabelEncoder()
x[:, 5] = labelEncoder_Embarked.fit_transform(x[:, 5])

#checkX = pd.DataFrame(x)
#
#oneHotCodeEncoder = OneHotEncoder(categorical_features=[5])
#x = oneHotCodeEncoder.fit_transform(x).toarray() 
#
#x[:,0] = np.ones(x[:,0].size).astype(int) 

#x_opt = x[:,[0,2,3,4,5,6]]
#regr_OLS = sm.OLS(endog = y,exog = x_opt).fit() 
#regr_OLS.summary()

#dfTest = pd.read_csv('/home/agl-android/.ML/Titanic/data/test.csv')
dfTest = pd.read_csv('D:/MachineLearning/Project/Titanic/data/test.csv')
xTest =  dfTest[['Pclass','Sex','Age','SibSp','Parch','Embarked']].values

checkX = pd.DataFrame(xTest)

imputerTest = SimpleImputer(strategy="mean")
xTest[:, 2:3] = imputerTest.fit_transform(xTest[:, 2:3]).astype(dtype= int)


labelEncoder_SexTest = LabelEncoder()
xTest[:, 1] = labelEncoder_SexTest.fit_transform(xTest[:, 1])

labelEncoder_EmbarkedTest = LabelEncoder()
xTest[:, 5] = labelEncoder_EmbarkedTest.fit_transform(xTest[:, 5])

#checkX = pd.DataFrame(x)

#oneHotCodeEncoder = OneHotEncoder(categorical_features=[5])
#xTest = oneHotCodeEncoder.fit_transform(xTest).toarray() 
#
#xTest[:,0] = np.ones(xTest[:,0].size).astype(int) 


regr = tree.DecisionTreeClassifier()
regr.fit(x,y)
yPridict = regr.predict(xTest).astype(dtype = int) 


out = pd.DataFrame()
out['PassengerId'] = dfTest['PassengerId'].values
out['Survived'] = yPridict

