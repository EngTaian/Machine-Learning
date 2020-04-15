# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:45:54 2020

@author: taian
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#read database
data = pd.read_csv('../csv/adult_data.csv')

#read previsors
previsors = data.iloc[:, 0:14].values

#read class
d_class = data.iloc[:, 14].values

#creating dummy variables
labelEncoder = LabelEncoder()
previsors[:,1] = labelEncoder.fit_transform(previsors[:, 1])
previsors[:,3] = labelEncoder.fit_transform(previsors[:, 3])
previsors[:,5] = labelEncoder.fit_transform(previsors[:, 5])
previsors[:,6] = labelEncoder.fit_transform(previsors[:, 6])
previsors[:,7] = labelEncoder.fit_transform(previsors[:, 7])
previsors[:,8] = labelEncoder.fit_transform(previsors[:, 8])
previsors[:,9] = labelEncoder.fit_transform(previsors[:, 9])
previsors[:,13] = labelEncoder.fit_transform(previsors[:, 13])


oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsors = oneHotEncoder.fit_transform(previsors).toarray()

d_class = labelEncoder.fit_transform(d_class)

#scaling variables
standardScaler = StandardScaler()
previsors = standardScaler.fit_transform(previsors)

previsorsTest, previsorsTraining,classTest,classTraining = train_test_split(previsors, d_class, test_size=0.15, random_state=0)





