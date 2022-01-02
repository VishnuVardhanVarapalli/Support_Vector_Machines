import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

X,y = make_classification(n_samples=200,n_features=10,random_state=20)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 43)

Model = LinearSVC()
Model.fit(X_train, y_train)
print("Model Trained")

predictions = Model.predict(X_test)
print("predictions made")

print(type(predictions[0]))
print(predictions[:5])
print(type(y_test[0]))

Model1 = SVC()
Model1.fit(X_train, y_train)
predictions1 = Model1.predict(X_test)

print("accuracy is:"+str(round(metrics.accuracy_score(y_test, predictions)*100, 2)))
print("accuracy is:"+str(round(metrics.accuracy_score(y_test, predictions1)*100, 2)))