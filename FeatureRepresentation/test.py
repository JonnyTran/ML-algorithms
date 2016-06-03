import numpy as np
import matplotlib.pyplot as plt

from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding

X = np.random.rand(1000, 100) - 0.5
n_samples, n_features = X.shape

sr = KSVDSparseCoding(n_components=200, max_iter=150, verbose=1)
sr.fit(X)

errs = X.T - np.dot(sr.dictionary, sr.code)
sample_errs = [np.linalg.norm(errs[:, n]) for n in range(n_samples)]

print "max(sample_errs)", max(sample_errs)
print "err matrix norm", np.linalg.norm(errs, ord="fro")
plt.plot(sr.errors)
print sr.dictionary
print sr.code
