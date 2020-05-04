# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:28:32 2020

@author: taian
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

def load_data(filename):
    csv_path = '../csv/' + filename
    return pd.read_csv(csv_path)

df = load_data('adult_data.csv')

previsors = df.iloc[:, 0:14].values
classData = df.iloc[:, 14].values

labelEncoder = LabelEncoder()
#1,3,5,6,7,8,9,13
previsors[:, 1] = labelEncoder.fit_transform(previsors[:,1])
previsors[:, 3] = labelEncoder.fit_transform(previsors[:,3])
previsors[:, 5] = labelEncoder.fit_transform(previsors[:,5])
previsors[:, 6] = labelEncoder.fit_transform(previsors[:,6])
previsors[:, 7] = labelEncoder.fit_transform(previsors[:,7])
previsors[:, 8] = labelEncoder.fit_transform(previsors[:,8])
previsors[:, 9] = labelEncoder.fit_transform(previsors[:,9])
previsors[:, 13] = labelEncoder.fit_transform(previsors[:,13])

oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),[1,3,5,6,7,8,9,13])], remainder="passthrough")
previsors = oneHotEncoder.fit_transform(previsors).toarray()

standardScaler = StandardScaler()
previsors = standardScaler.fit_transform(previsors)

previsorTraining, previsorsTest, classTraining,classTest = train_test_split(previsors, classData,test_size=0.15, random_state=0)

classifier = KNeighborsClassifier()
classifier.fit(previsorTraining, classTraining)
result = classifier.predict(previsorsTest)

matrix = confusion_matrix(result, classTest)
accuracy = accuracy_score(result, classTest)