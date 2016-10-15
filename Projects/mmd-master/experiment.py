from mmd import mmd_approx, mmd_full

import theano.tensor as T
import theano
import numpy as np


class NaiveExperiment(object):
    def __init__(self, target_distribution, initial_distribution, batch_size=100, alpha=1.0, objective=mmd_approx,
                 num_kernels=5):
        self.target = target_distribution
        self.actual = initial_distribution
        self.batch_size = batch_size
        self.alpha = alpha
        self.objective = objective

        x = T.dmatrix()
        y = T.dmatrix()
        mmd_fun = self.objective
        # In the paper rather than trying to decide on the parameter sigma, they use a mixture of 5 kernels with varying sigmas
        mmd_objective = mmd_fun(x, y) + mmd_fun(x, y, 0.1) + mmd_fun(x, y, 1.0) + mmd_fun(x, y, 1.5) + mmd_fun(x, y,
                                                                                                               2.0)
        grad = T.grad(mmd_objective, wrt=y)
        self.train_function = theano.function([x, y], [mmd_objective, grad])

    def update(self):
        indices_x = np.random.choice(range(self.target.shape[1]), size=(self.batch_size,), replace=False)
        indices_y = np.random.choice(range(self.actual.shape[1]), size=(self.batch_size,), replace=False)

        # Compute approximate MMD on these two samples
        mmd_loss, gradient = self.train_function(self.target[:, indices_x], self.actual[:, indices_y])
        # Update the values in y that were sampled in the negative direction of the gradient
        self.actual[:, indices_y] -= self.alpha * (gradient / np.linalg.norm(gradient))


def default_experiment():
    from util import bimodal_random
    sample_size = 10000
    x = bimodal_random(sample_size).reshape(1, -1)
    # Generate uniform random numbers between -3 and 3
    y = np.random.uniform(-3, 3, (1, sample_size))

    exp = NaiveExperiment(x, y, alpha=3.0, batch_size=5000)
    return exp
