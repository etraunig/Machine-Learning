import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        #get distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #get k nearest samples
        k_indices = np.argsort(distances)[:self.k]

        #get the labels of the k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #TODO: return the most common class label