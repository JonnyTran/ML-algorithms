import logging
import time

import numpy as np
from pylab import *
from sklearn.linear_model import orthogonal_mp_gram, LassoLars


class SparseRepresentation():
    def __init__(self, n_components=None, alpha=1, max_iter=1000,
                 fit_algorithm='lars', transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=1,
                 n_jobs=1, verbose=False):

        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm

        self.transform_alpha = transform_alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.verbose = verbose

    def fit(self, X, y=None):
        self.initialize(X)

        self.dict_learning(X, self.n_components, self.alpha, max_iter=100, tol=1e-8,
                           method='lars', n_jobs=1, dict_init=None, code_init=None,
                           callback=None, verbose=False, random_state=None,
                           return_n_iter=False)

    def initialize(self, X):
        self.n_samples, self.n_features = X.shape
        self.dictionary = np.random.rand(self.n_features, self.n_components)
        self.code = np.random.rand(self.n_components, self.n_samples)

    def dict_learning(self, X, n_components, alpha, max_iter=100, tol=1e-8,
                      method='omp', n_jobs=1, dict_init=None, code_init=None,
                      callback=None, verbose=False, random_state=None,
                      return_n_iter=False):

        t0 = time.time()
        for i in range(max_iter):
            dt = (time.time() - t0)

            # Update code
        self.code = self.sparse_encode(X, self.dictionary, algorithm=method, alpha=alpha,
                                       init=self.code, n_jobs=n_jobs)

        self.dictionary = self.update_dict(self.dictionary.T, X.T, self.code.T,
                                           verbose=verbose)

    def sparse_encode(self, X, dictionary, gram, cov=None, algorithm='lasso_lars',
                      regularization=None, copy_cov=True,
                      init=None, max_iter=1000, check_input=True, verbose=0):
        if algorithm == 'omp':
            new_code = orthogonal_mp_gram(
                Gram=gram, Xy=cov, n_nonzero_coefs=int(regularization),
                tol=None, norms_squared=row_norms(X, squared=True),
                copy_Xy=copy_cov).T

        elif algorithm == 'lasso':
            alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all='ignore')

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False,
                                   verbose=verbose, normalize=False,
                                   precompute=gram, fit_path=False)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)

    def update_dict(self, dictionary, X, code, verbose):
        print "hello"

    def ksvd(components_, K, T, D=None, max_err=0, max_iter=10, approx=False, preserve_dc=False):
        logger = logging.getLogger(__name__)

        (N, M) = components_.shape

        # if we're not given a dictionary for starters, make our own
        if D is None:
            D = rand(N, K)

        # make the first dictionary element constant; remove the mean from the
        # rest of the dictionary elements
        if preserve_dc:
            D[:, 0] = 1
        for i in range(1, K): D[:, i] -= mean(D[:, i])

        # normalize the dictionary regardless
        for i in range(K):
            D[:, i] /= norm(D[:, i])

        logger.info("running ksvd on %d %d-dimensional vectors with K=%d" \
                    % (M, N, K))

        # algorithm stuff
        X = zeros((K, N))
        err = inf
        iter_num = 0

        while iter_num < max_iter and err > max_err:
            # batch omp, woo!
            logger.info("staring omp...")
            # X = omp(D, Y, T, max_err)
            logger.info("omp complete!")
            logger.info( \
                'average l0 "norm" for ksvd iteration %d after omp was %f' \
                % (iter_num, len(nonzero(X)[0]) / M))

            # dictionary update -- protip: update dictionary columns in random
            # order
            atom_indices = range(K)
            if preserve_dc: atom_indices = atom_indices[1:]
            random.shuffle(atom_indices)

            unused_atoms = []

        for (i, j) in zip(atom_indices, xrange(K)):
            if False:
                if j % 25 == 0:
                    logger.info("ksvd: iteration %d, updating atom %d of %d" \
                                % (iter_num + 1, j, K))

            # find nonzero entries
            x_using = nonzero(X[i, :])[0]

            if len(x_using) == 0:
                unused_atoms.append(i)
                continue

            if not approx:
                # Non-approximate K-SVD, as described in the original K-SVD
                # paper

                # compute residual error ... here's a trick passing almost all the
                # work to BLAS
                X[i, x_using] = 0
                Residual_err = components_[:, x_using] - dot(D, X[:, x_using])

                # update dictionary and weights -- sparsity-restricted rank-1
                # approximation
                U, s, Vt = svd(Residual_err)
                D[:, i] = U[:, 0]
                X[i, x_using] = s[0] * Vt.T[:, 0]
            else:
                # Approximate K-SVD

                D[:, i] = 0

                g = X[i, x_using]
                d = dot(components_[:, x_using], g) - dot(D[:, x_using], g)
                d = d / norm(d)
                g = dot(components_[:, x_using].T, d) - dot(D[:, x_using].T, d)

                D[:, i] = d
                X[i, x_using] = g

        # fill in values for unused atoms

        # unused column -> replace by signal in training data with worst
        # representation
        Repr_err = components_ - dot(D, X)
        Repr_err_norms = (norm(Repr_err[:, n]) for n in range(M))

        err_indices = sorted(zip(Repr_err_norms, xrange(M)), reverse=True)

        for (unused_index, err_tuple) in zip(unused_atoms, err_indices):
            (err, err_idx) = err_tuple

            d = components_[:, err_idx].copy()
            if preserve_dc: d -= mean(d)
            d /= norm(d)
            D[:, unused_index] = d

        # compute maximum representation error
        Repr_err_norms = [norm(Repr_err[:, n]) for n in range(M)]
        err = max(Repr_err_norms)

        # and increase the iter_num; repeat
        iter_num += 1

        # report a bit of info
        logger.info( \
            "maximum representation error for ksvd iteration %d was %f" \
            % (iter_num, err))
