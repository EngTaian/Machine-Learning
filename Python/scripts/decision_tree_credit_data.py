import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv('../csv/credit_data.csv')
meanAge = base['age'][base['age'] > 0].mean()
base.loc[base.age <0, 'age'] = meanAge
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]


previsors = base.iloc[:, 1:4].values
b_class = base.iloc[:, 4].values


imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
imputer = imputer.fit(previsors[:, 0:3])
previsors[:,0:3] = imputer.transform(previsors[:, 0:3])


scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)

previsorsTest, previsorsTraining,classTest,classTraining = train_test_split(previsors, b_class, test_size=0.25, random_state=0)


tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
tree.fit(previsorsTraining, classTraining)
result = tree.predict(previsorsTest)

matrix = confusion_matrix(classTest, result)
accuracy = accuracy_score(classTest, result)
