# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:49:14 2020

@author: taian
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

def load_data(filename):
    csv_path = '../csv/' + filename
    return pd.read_csv(csv_path)

database = load_data('credit_data.csv')
database.loc[database['age'] < 0, 'age'] = database.loc[database['age'] > 0, 'age'].mean()

previsors = database.iloc[:, 0:4].values
classData = database.iloc[:,4].values

simpleImputer =SimpleImputer(missing_values=np.nan, strategy='mean')
simpleImputer = simpleImputer.fit(previsors[:,0:3])
previsors[:,0:3] = simpleImputer.fit_transform(previsors[:,0:3])

standardScaler = StandardScaler()
previsors = standardScaler.fit_transform(previsors)

previsorTraining, previsorsTest,classTraining, classTest = train_test_split(previsors, classData,test_size=0.25, random_state=0)

classifier = KNeighborsClassifier()
classifier.fit(previsorTraining, classTraining)
result = classifier.predict(previsorsTest)

matrix = confusion_matrix(result, classTest)
accuracy = accuracy_score(result, classTest)