import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


class TreeSparseRepresentation:
    class DictionaryNode:
        def __init__(self, outer, parent=None):
            self.outer = outer
            self.parent = parent
            self.children = []

            self.dictionary = np.random.rand(self.outer.n_features, self.outer.n_children_per_node) - 0.5
            self.dictionary_counts = None

        def dict_learning(self, X_residual):
            pass

        def sparse_encode(self, X_residual):
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.outer.n_nonzero_coefs)
            omp.fit(self.dictionary, X_residual.T)
            new_code = omp.coef_.T

            return new_code

        def update_dict(self):
            pass

        def count_sample_by_class(self):
            # TODO Count samples used by each dictionary atom
            pass

        def count_sample_by_atom(self):
            # TODO Count samples used by each dictionary atom
            pass

        def get_residual(self):
            # TODO compute residual to pass to childnode
            pass

        def add_child(self, child):
            self.children.append(child)

        def inference(self, X):
            # TODO Go down the tree to select next best dictionary atoms
            pass

    def __init__(self, n_children_per_node=5, n_nonzero_coefs=2, min_obj_per_node=10, max_iter=50):
        self.n_children_per_node = n_children_per_node
        self.n_nonzero_coefs = n_nonzero_coefs
        self.min_obj = min_obj_per_node
        self.max_iter = max_iter

    def fit(self, X, y):
        self.initialize(X, y)

    def initialize(self, X, y):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.dictionary = self.DictionaryNode(self, parent=None)


        # Prints
        print "X: n_samples", self.n_samples, ", n_features", self.n_features

    def sparse_encode(self, X):
        """
        Select dictionary atoms down the tree with greedy Orthogonal Matching Pursuit
        :param X:
        """
        pass

    def next_active_set(self, residual_node, dict_node, code_node):
        pass
