import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('train.csv')
num_rows = df.shape[0]
y = df['price_range']

x = df.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_std,y, test_size=0.2)

svc = svm.SVC()
parameters = {'kernel':('linear','rbf'), 'C':[1,10]}
clf=GridSearchCV(svc,parameters)
model = clf.fit(X_train,y_train)
model.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)


data_test = pd.read_csv('test.csv')
y = data_test.ix[:,:-1].values
standard_scaler1 = StandardScaler()
x_std1 = standard_scaler1.fit_transform(y)
prediction = clf.predict(x_std1)
np.savetxt('predicted_set.csv',prediction, delimiter=',')
pred=clf.predict(X_test)
