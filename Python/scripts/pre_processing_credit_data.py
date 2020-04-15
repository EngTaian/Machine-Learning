
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#read database
base = pd.read_csv('../csv/credit_data.csv')
base.describe()
#find age data less than 0
print(base.loc[base['age'] < 0])
#delete age data less than 0
#base.drop(base[base.age < 0].index, inplace=True)
#global mean
base.mean()
#age mean without zero
meanAge = base['age'][base['age'] > 0].mean()
#define meanAge to age less than zero into base
base.loc[base.age <0, 'age'] = meanAge
#find null value in age
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

#creating previsors and class
previsors = base.iloc[:, 1:4].values
b_class = base.iloc[:, 4].values

#
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
imputer = imputer.fit(previsors[:, 0:3])
previsors[:,0:3] = imputer.transform(previsors[:, 0:3])

#scaling variables
scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)

#creating test and training database
previsorsTest, previsorsTraining,classTest,classTraining = train_test_split(previsors, b_class, test_size=0.25, random_state=0)
