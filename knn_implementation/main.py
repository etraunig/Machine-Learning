# This code was made using copilot for formatting and lib searching

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

knn = KNN(5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print("accuracy:", accuracy)