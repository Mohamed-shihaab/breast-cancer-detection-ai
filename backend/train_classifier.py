import numpy as np
from classifier import BreastCancerClassifier

# Dummy dataset (simulate extracted features)
X = np.array([
    [0.2, 0.1],
    [0.3, 0.25],
    [0.8, 0.9],
    [0.75, 0.85]
])

# Labels: 0 = benign, 1 = malignant
y = np.array([0, 0, 1, 1])

clf = BreastCancerClassifier()
clf.train(X, y)

pred = clf.predict(X)
print("Predictions:", pred)
