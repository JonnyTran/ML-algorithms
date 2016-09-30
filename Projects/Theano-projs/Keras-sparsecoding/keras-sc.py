import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

import theano
import theano.tensor as T

import keras.initializations as initializations
import keras.activations as activations
from keras.layers.core import Layer
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop


# from seya.layers.coding import SparseCoding

# from agnez import grid2d

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val))


def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)


def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](0.), *dims)


def _IstaStep(cost, states, lr=.001, lambdav=.1, x_prior=0):
    grads = T.grad(cost, states)
    new_x = states - lr * grads
    if x_prior != 0:
        new_x += lambdav * lr * .1 * _proxInnov(states, x_prior)
    new_states = _proxOp(new_x, lr * lambdav)
    return theano.gradient.disconnected_grad(new_states)


def _RMSPropStep(cost, states, accum_1, accum_2):
    rho = .9
    lr = .009
    momentum = .9
    epsilon = 1e-8

    grads = T.grad(cost, states)

    new_accum_1 = rho * accum_1 + (1 - rho) * grads ** 2
    new_accum_2 = momentum * accum_2 - lr * grads / T.sqrt(new_accum_1 + epsilon)
    denominator = T.sqrt(new_accum_1 + epsilon)
    new_states = states + momentum * new_accum_2 - lr * (grads / denominator)
    new_states = _proxOp(states - lr * (grads / denominator),
                         .1 * lr / denominator) + momentum * new_accum_2
    return (theano.gradient.disconnected_grad(new_states),
            new_accum_1, new_accum_2)


def _proxOp(x, t):
    return T.maximum(x - t, 0) + T.minimum(x + t, 0)


def _proxInnov(x, x_tm1):
    innov = x - x_tm1
    i0 = T.maximum(innov, 1)
    i1 = T.minimum(i0, -1)
    return i1


class SparseCoding(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform',
                 activation='linear',
                 truncate_gradient=-1,
                 gamma=.1,
                 n_steps=10,
                 return_reconstruction=False,
                 W_regularizer=None,
                 activity_regularizer=None, **kwargs):

        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_reconstruction = return_reconstruction
        self.input = T.matrix()

        self.W = self.init((self.output_dim, self.input_dim))
        self.trainable_weights = [self.W, ]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

        kwargs['input_shape'] = (self.input_dim,)
        super(SparseCoding, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_reconstruction:
            return input_shape
        else:
            return input_shape[0], self.ouput_dim

    def build(self):
        pass

    def get_initial_states(self, X):
        return alloc_zeros_matrix(X.shape[0], self.output_dim)

    def _step(self, x_t, inputs, prior, W):
        outputs = self.activation(T.dot(x_t, self.W))
        rec_error = T.sqr(inputs - outputs).sum()
        x = _IstaStep(rec_error, x_t, lambdav=self.gamma, x_prior=prior)
        return x, outputs

    def _get_output(self, inputs, train=False, prior=0):
        initial_states = self.get_initial_states(inputs)
        outputs, updates = theano.scan(
            self._step,
            sequences=[],
            outputs_info=[initial_states, None],
            non_sequences=[inputs, prior, self.W],
            n_steps=self.n_steps,
            truncate_gradient=self.truncate_gradient)

        outs = outputs[0][-1]
        if self.return_reconstruction:
            return T.dot(outs, self.W)
        else:
            return outs

    def get_output(self, train=False):
        inputs = self.get_input(train)
        return self._get_output(inputs, train)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "return_reconstruction": self.return_reconstruction}


def main():
    S = loadmat('patches.mat')['data'].T.astype(theano.config.floatX)
    print S.shape

    image_patches = fetch_mldata("natural scenes data")
    X = image_patches.data

    mean = S.mean(axis=0)
    S -= mean[np.newaxis]

    model = Sequential()
    model.add(
        SparseCoding(
            input_dim=256,
            output_dim=1000,  # we are learning 49 filters,
            n_steps=25,  # remember the self.n_steps in the scan loop?
            truncate_gradient=1,  # no backpropagation through time today now,
            # just regular sparse coding
            W_regularizer=l2(.00005),
            return_reconstruction=True  # we will output Ax which approximates the input
        )
    )

    rmsp = RMSprop(lr=.1)
    model.compile(loss='mse', optimizer=rmsp)  # RMSprop for Maximization as well

    nb_epoch = 100
    batch_size = 100
    model.fit(S,  # input
              S,  # and output are the same thing, since we are doing generative modeling.
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              show_accuracy=False,
              verbose=2)

    A = model.params[0].get_value()
    I = grid2d(A)
    plt.imshow(I)


if __name__ == "__main__":
    main()
