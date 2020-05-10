# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:13:16 2020

@author: taian
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv('../csv/adult_data.csv')

previsors = base.iloc[:, 0:14].values
classe = base.iloc[:,14].values

labelEncoder = LabelEncoder()
previsors[:,1] = labelEncoder.fit_transform(previsors[:,1])
previsors[:,3] = labelEncoder.fit_transform(previsors[:,3])
previsors[:,5] = labelEncoder.fit_transform(previsors[:,5])
previsors[:,6] = labelEncoder.fit_transform(previsors[:,6])
previsors[:,7] = labelEncoder.fit_transform(previsors[:,7])
previsors[:,8] = labelEncoder.fit_transform(previsors[:,8])
previsors[:,9] = labelEncoder.fit_transform(previsors[:,9])
previsors[:,13] = labelEncoder.fit_transform(previsors[:,13])

oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsors = oneHotEncoder.fit_transform(previsors).toarray()

standardScaler = StandardScaler()
#102,103,104,105,106,107
previsors = standardScaler.fit_transform(previsors)

#separe base in training and test
previsorsTraining, previsorsTest, classTraining, classTest = train_test_split(previsors, classe, test_size=0.15, random_state=0)

classifier = LogisticRegression(random_state=1)
classifier = classifier.fit(previsorsTraining, classTraining)
result = classifier.predict(previsorsTest)

#metrics
precision = accuracy_score(classTest, result)
matrix = confusion_matrix(classTest, result)
