import numpy as np

from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding

X = np.random.rand(100, 100)
n_samples, n_features = X.shape

sr = KSVDSparseCoding(n_components=200, max_iter=300, verbose=1)
sr.fit(X)

errs = X.T - np.dot(sr.dictionary, sr.code)
sample_errs = [np.linalg.norm(errs[:, n]) for n in range(n_samples)]
print max(sample_errs)
print np.linalg.norm(errs, ord="fro")
