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

df = pd.read_csv('/home/agl-android/.ML/Titanic/data/train.csv')

x = df[['PassengerId','Pclass','Sex','Age']].values
y = df[['Survived']].values

imputer = SimpleImputer(strategy="mean")
x[:, 3:4] = imputer.fit_transform(x[:, 3:4]).astype(dtype= int)

labelEncoder_X = LabelEncoder()
x[:, 2] = labelEncoder_X.fit_transform(x[:, 2])

