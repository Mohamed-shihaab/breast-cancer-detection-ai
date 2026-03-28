import numpy as np
from sklearn.decomposition import PCA

def reduce_features(features, n_components=8):
    # simulate dataset (for demo)
    X = np.vstack([features, features])

    max_components = min(X.shape[0], X.shape[1])

    pca = PCA(n_components=max_components)
    reduced = pca.fit_transform(X)

    return reduced[0]
