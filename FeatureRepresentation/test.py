import numpy as np

from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding

X = np.random.rand(50, 40)
n_samples, n_features = X.shape

sr = KSVDSparseCoding(n_components=100, verbose=True)
sr.fit(X)
