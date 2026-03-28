import numpy as np
from sklearn.svm import SVC

class BreastCancerClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
