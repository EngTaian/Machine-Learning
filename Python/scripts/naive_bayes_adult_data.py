import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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

oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),[1,2,5,6,7,8,9,13])], remainder="passthrough")
previsors = oneHotEncoder.fit_transform(previsors).toarray()

