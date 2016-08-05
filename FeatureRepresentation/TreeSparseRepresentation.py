import numpy as np
import time
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_classification
from sklearn import preprocessing


class TreeSparseRepresentation:
    class DictionaryNode:
        def __init__(self, outer, parent=None):
            self.outer = outer
            self.parent = parent
            self.children = []

        def fit(self, X_subset, X_indices):
            """

            :param X: Subset of the samples
            :param X_indices: the indicies of samples to be fitted in this dictionary node
            """
            self.X_residual = X_subset
            self.X_indicies = X_indices
            self.n_samples, self.n_features = self.X_residual.shape

            # Initialize dictionary
            self.dictionary = np.random.rand(self.outer.n_features, self.outer.n_components_per_node) - 0.5
            self.dictionary_counts = None

            print "\nDictionary Node"
            print "X: n_samples", self.n_samples, ", X_indicies:", self.X_indicies, ", n_features", self.n_features
            print "dictionary", self.dictionary.shape

            self.dict_learning()
            print "atom_bin_count", self.atom_bin_count()

        def dict_learning(self):
            t0 = time.time()
            iter = 0

            while iter < self.outer.max_iter:
                iter += 1
                # Update code with fixed dictionary
                self.code = self.sparse_encode(self.outer.n_nonzero_coefs)

                # Update dictionary with fixed code
                self.update_dict()

                repr_err = self.X_residual.T - np.dot(self.dictionary, self.code)
                repr_err_norms = [np.linalg.norm(repr_err[:, n]) for n in range(self.n_samples)]
                err = np.mean(repr_err_norms)
                dt = time.time() - t0
                if self.outer.verbose:
                    print("K-SVD iteration %d, % 3is elapsed, MSE: %f"
                          % (iter, dt, err))

            self.X_residual = (self.X_residual.T - np.dot(self.dictionary, self.code)).T

        def sparse_encode(self, n_nonzero_coefs):
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
            omp.fit(self.dictionary, self.X_residual.T)
            new_code = omp.coef_.T

            return new_code

        def update_dict(self):
            (self.n_features, self.n_components) = self.dictionary.shape
            atom_indices = range(self.n_components)
            np.random.shuffle(atom_indices)

            unused_atoms = []

            # Iterating through every dictionary atoms in random order
            for (i, j) in zip(atom_indices, xrange(self.n_components)):
                if self.outer.verbose >= 2:
                    if j % 25 == 0:
                        print("ksvd: updating atom %d of %d" \
                              % (j, self.n_components))

                x_using = np.nonzero(self.code[i, :])[0]

                if len(x_using) == 0:
                    unused_atoms.append(i)
                    continue

                # Normal K-SVD with running svd step on residual error iterating through every dictionary columns
                self.code[i, x_using] = 0
                residual_err = self.X_residual[x_using, :].T - np.dot(self.dictionary, self.code[:, x_using])

                U, s, V = np.linalg.svd(residual_err)
                self.dictionary[:, i] = U[:, 0]
                self.code[i, x_using] = s[0] * V.T[:, 0]

        def atom_bin_count_by_class(self):
            # TODO Count samples used by each dictionary atom grouped by class label
            pass

        def atom_bin_count(self):
            # Count no of samples used by each dictionary atom
            return np.bincount(self.code.nonzero()[0])

        def sample_bin_count(self):
            # Count no of dictionary atoms used by each sample
            return np.bincount(self.code.nonzero()[1])

        def x_using(self, atom_idx):
            # Return indices of samples that uses this dictionary atom
            return np.nonzero(self.code[atom_idx, :])[0]

        def x_using_classes(self, atom_idx):
            # Count class distribution of samples that uses this dictionary atom
            a = np.nonzero(self.code[atom_idx, :])[0]  # samples in indicies starting from 0
            b = self.X_indicies[a]  # samples with real indicies from dataset
            c = self.outer.Y[b]  # class label of b
            return np.bincount(c)

        def add_child(self, child):
            self.children.append(child)

        def inference(self, X):
            # TODO Go down the tree to select next best dictionary atoms
            pass

    def __init__(self, n_components_per_node=5, n_nonzero_coefs=2, min_obj_per_node=10, max_iter=50, verbose=1):
        self.n_components_per_node = n_components_per_node
        self.n_nonzero_coefs = n_nonzero_coefs
        self.min_obj = min_obj_per_node
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, Y):
        self.initialize(X, Y)

        self.dictionary.fit(X, range(self.n_samples))

    def initialize(self, X, Y):
        self.X = X
        self.Y = Y
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


def main():
    X, y = make_classification(n_samples=100, n_features=20, n_redundant=0, n_classes=2, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    X = preprocessing.scale(X)
    print y
    n_samples, n_features = X.shape

    tsr = TreeSparseRepresentation(n_components_per_node=50, n_nonzero_coefs=3, max_iter=15, verbose=1)
    tsr.fit(X, y)


    # errs = X.T - np.dot(sr.dictionary, sr.code)
    # sample_errs = [np.linalg.norm(errs[:, n]) for n in range(n_samples)]

    # print "max(sample_errs)", max(sample_errs)
    # print "err matrix norm", np.linalg.norm(errs, ord="fro") / n_samples
    # print sr.dictionary
    # print sr.code


if __name__ == "__main__":
    main()
