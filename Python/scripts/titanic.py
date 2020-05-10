# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:13:07 2020

@author: taian
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def load_data(filename):
    csv_path='../csv/'+filename
    return pd.read_csv(csv_path)

df_train = load_data('titanic_train.csv')
df_test = load_data('titanic_test.csv')

features=['Pclass','Sex', 'Age', 'SibSp','Parch']
df_previsors = df_train[features]
df_class = df_train['Survived']

simpleFeature=['Pclass', 'Age', 'SibSp','Parch']

simpleImputer = SimpleImputer(missing_values=np.nan, strategy="mean")
simpleImputer=simpleImputer.fit(df_previsors[simpleFeature])
df_previsors[simpleFeature] = simpleImputer.transform(df_previsors[simpleFeature])

df_previsors = pd.get_dummies(df_previsors[features])

previsors_test=df_test[features]
simpleImputertTest= SimpleImputer(missing_values=np.nan, strategy="mean")
simpleImputertTest=simpleImputer.fit(previsors_test[simpleFeature])
previsors_test[simpleFeature] = simpleImputertTest.transform(previsors_test[simpleFeature])

previsors_test= pd.get_dummies(previsors_test[features])

classification = RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0)
classification = classification.fit(df_previsors, df_class)
prediction = classification.predict(previsors_test)

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': prediction})
output.to_csv('../csv/result.csv', index=False)


