# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:33:51 2020

@author: taian
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(filename):
    csv_path = '../csv/'+filename
    return pd.read_csv(csv_path)

base = load_data('credit_data.csv')
base.loc[base['age'] < 0, 'age'] = base['age'][base['age'] > 0].mean()

previsors = base.iloc[:, 1:4].values
b_class = base.iloc[:, 4].values

imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
imputer = imputer.fit(previsors[:, 0:3])
previsors[:,0:3] = imputer.transform(previsors[:, 0:3])


scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)

previsorsTest, previsorsTraining,classTest,classTraining = train_test_split(previsors, b_class, test_size=0.25, random_state=0)

classifier = RandomForestClassifier(n_estimators=15,criterion='entropy',random_state=0)
classifier.fit(previsorsTraining, classTraining)
prevision = classifier.predict(previsorsTest)

matrix = confusion_matrix(prevision, classTest)
accuracy= accuracy_score(prevision,classTest)