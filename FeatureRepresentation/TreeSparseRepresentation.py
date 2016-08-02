import numpy as np


class TreeSparseRepresentation:
    def __init__(self, n_children=5, min_obj=10, max_iter=50):
        self.n_children = n_children
        self.min_obj = min_obj
        self.max_iter = max_iter

    def fit(self, X, y):
        self.initialize(X, y)

    def initialize(self, X, y):
        self.X = X
        self.n_samples, self.n_features = X.shape
        # self.dictionary = np.random.rand(self.n_features, self.n_components) - 0.5
        self.code = np.random.rand(self.n_components, self.n_samples) - 0.5

        # Prints
        print "X: n_samples", self.n_samples, ", n_features", self.n_features
        print "dictionary", self.dictionary.shape
        print "code", self.code.shape
