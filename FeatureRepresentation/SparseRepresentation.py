import time

import pylab as pl
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram, LassoLars, OrthogonalMatchingPursuit
from sklearn.utils.extmath import row_norms


class KSVDSparseCoding():
    def __init__(self, n_components=None, alpha=1, max_iter=1000,
                 transform_n_nonzero_coefs=None, transform_alpha=1,
                 n_jobs=1, verbose=False):

        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter

        self.transform_alpha = transform_alpha
        self.verbose = verbose

    def fit(self, X, y=None):
        self.initialize(X)

        self.dict_learning(X, self.n_components, self.alpha, max_iter=100, tol=1e-8,
                           dict_init=None, code_init=None,
                           verbose=self.verbose)

    def initialize(self, X):
        """

        :param X: Expecting an numpy array of order (n_samples x n_features)
        """
        self.X = X.T
        self.n_features, self.n_samples = X.shape
        self.dictionary = np.random.rand(self.n_features, self.n_components)
        self.code = np.random.rand(self.n_components, self.n_samples)

        # Prints
        print "X: n_samples", self.n_samples, ", n_features", self.n_features
        print "dictionary", self.dictionary.shape
        print "code", self.code.shape

    def dict_learning(self, X, n_components, alpha, max_iter=100, tol=1e-8,
                      method='omp', dict_init=None, code_init=None, verbose=True):

        t0 = time.time()
        iter = 0
        while iter < max_iter:
            iter += 1
            dt = (time.time() - t0)

            # Update code
            self.code = self.sparse_encode(X, self.dictionary, alpha=alpha)

            # Update dictionary
            self.dictionary, unused_atoms = self.update_dict(self.dictionary, self.X, self.code,
                                           verbose=verbose)

            # Fill in values for unused atoms by worst reconstructed samples
            repr_err = X - np.dot(self.dictionary, self.code)
            repr_err_norms = (np.linalg.norm(repr_err[:, n]) for n in range(self.n_features))
            err_indices = sorted(zip(repr_err_norms, xrange(self.n_features)), reverse=True)

            for (unused_index, err_tuple) in zip(unused_atoms, err_indices):
                (err, err_idx) = err_tuple

                d = X[:, err_idx].copy()
                d -= np.mean(d)
                d /= np.linalg.norm(d)
                self.dictionary[:, unused_index] = d

            repr_err_norms = [np.linalg.norm(repr_err[:, n]) for n in range(self.n_features)]
            err = max(repr_err_norms)

            if verbose:
                print("K-SVD iteration %d, % 4.1fmn elapsed, maximum representation error: %f"
                      % (iter, dt, err))

    def sparse_encode(self, X, dictionary, alpha, max_iter=1000, verbose=0):

        omp = OrthogonalMatchingPursuit()
        omp.fit(dictionary, X)
        new_code = omp.coef_.T

        return new_code

    def update_dict(self, dictionary, X, code, verbose):
        (n_features, n_components) = dictionary.shape

        # Normalize and zero mean dictionary atoms
        for i in range(n_components):
            dictionary[:, i] -= np.mean(dictionary[:, i])
            dictionary[:, i] /= np.linalg.norm(dictionary[:, i])

        atom_indices = range(n_components)
        np.random.shuffle(atom_indices)

        unused_atoms = []

        # Iterating through every dictionary atoms in random order
        for (i, j) in zip(atom_indices, xrange(n_components)):
            if verbose:
                # if j % 25 == 0:
                print("ksvd: updating atom %d (%d) of %d" \
                      % (i, j, n_components))

            print "code[i, :].shape", code[i, :].shape
            x_using = np.nonzero(code[i, :])[0]
            print "x_using", x_using

            if len(x_using) == 0:
                unused_atoms.append(i)
                continue

            code[i, x_using] = 0
            residual_err = X[:, x_using] - np.dot(dictionary, code[:, x_using])

            U, s, Vt = np.linalg.svd(residual_err)
            dictionary[:, i] = U[:, 0]
            code[i, x_using] = s[0] * Vt.T[:, 0]

        return dictionary, unused_atoms

    def ksvd(self, X, n_components, dictionary=None, max_err=0, max_iter=10, approx=False, preserve_dc=False):
        (n_samples, n_features) = X.shape

        # if we're not given a dictionary for starters, make our own
        if dictionary is None:
            dictionary = np.random.rand(n_samples, n_components)

        # make the first dictionary element constant; remove the mean from the
        # rest of the dictionary elements
        if preserve_dc:
            dictionary[:, 0] = 1
        for i in range(1, n_components): dictionary[:, i] -= np.mean(dictionary[:, i])

        # normalize the dictionary regardless
        for i in range(n_components):
            dictionary[:, i] /= np.linalg.norm(dictionary[:, i])

        print("running ksvd on %d %d-dimensional vectors with K=%d" \
              % (n_features, n_samples, n_components))

        # algorithm stuff
        code = np.zeros((n_components, n_samples))
        err = np.inf
        iter_num = 0

        while iter_num < max_iter and err > max_err:
            # batch omp, woo!
            print("staring omp...")
            # X = omp(dictionary, Y, T, max_err)
            print("omp complete!")
            print( \
                'average l0 "norm" for ksvd iteration %d after omp was %f' \
                % (iter_num, len(np.nonzero(code)[0]) / n_features))

            # dictionary update -- protip: update dictionary columns in random
            # order
            atom_indices = range(n_components)
            if preserve_dc: atom_indices = atom_indices[1:]
            np.random.shuffle(atom_indices)

            unused_atoms = []

            for (i, j) in zip(atom_indices, xrange(n_components)):
                if False:
                    if j % 25 == 0:
                        print("ksvd: iteration %d, updating atom %d of %d" \
                              % (iter_num + 1, j, n_components))

                # find nonzero entries
                x_using = np.nonzero(code[i, :])[0]

                if len(x_using) == 0:
                    unused_atoms.append(i)
                    continue

                if not approx:
                    # Non-approximate K-SVD, as described in the original K-SVD
                    # paper

                    # compute residual error ... here's a trick passing almost all the
                    # work to BLAS
                    code[i, x_using] = 0
                    Residual_err = X[:, x_using] - np.dot(dictionary, code[:, x_using])

                    # update dictionary and weights -- sparsity-restricted rank-1
                    # approximation
                    U, s, Vt = pl.svd(Residual_err)
                    dictionary[:, i] = U[:, 0]
                    code[i, x_using] = s[0] * Vt.T[:, 0]
                else:
                    # Approximate K-SVD

                    dictionary[:, i] = 0

                    g = code[i, x_using]
                    d = np.dot(X[:, x_using], g) - np.dot(dictionary[:, x_using], g)
                    d = d / np.linalg.norm(d)
                    g = np.dot(X[:, x_using].T, d) - np.dot(dictionary[:, x_using].T, d)

                    dictionary[:, i] = d
                    code[i, x_using] = g

            # fill in values for unused atoms

            # unused column -> replace by signal in training data with worst
            # representation
            Repr_err = X - np.dot(dictionary, code)
            Repr_err_norms = (np.linalg.norm(Repr_err[:, n]) for n in range(n_features))

            err_indices = sorted(zip(Repr_err_norms, xrange(n_features)), reverse=True)

            for (unused_index, err_tuple) in zip(unused_atoms, err_indices):
                (err, err_idx) = err_tuple

                d = X[:, err_idx].copy()
                if preserve_dc: d -= np.mean(d)
                d /= np.linalg.norm(d)
                dictionary[:, unused_index] = d

            # compute maximum representation error
            Repr_err_norms = [np.linalg.norm(Repr_err[:, n]) for n in range(n_features)]
            err = max(Repr_err_norms)

            print("maximum representation error: %f" % (err))
