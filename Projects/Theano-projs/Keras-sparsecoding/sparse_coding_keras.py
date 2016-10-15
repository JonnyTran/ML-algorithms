import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

import theano
import theano.tensor as T

from keras.layers.core import Layer
from keras import backend as K
from keras import activations, initializations

from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop

from agnez import grid2d


def _proxOp(x, t):
    return T.maximum(x - t, 0) + T.minimum(x + t, 0)


def _proxInnov(x, x_tm1):
    innov = x - x_tm1
    i0 = T.maximum(innov, 1)
    i1 = T.minimum(i0, -1)
    return i1


def shared_zeros(shape, dtype=theano.config.floatX, name='', n=1):
    shape = shape if n == 1 else (n,) + shape
    return theano.shared(np.zeros(shape, dtype=dtype), name=name)


def _IstaStep(cost, states, lr=.001, lambdav=.1, x_prior=0):
    grads = T.grad(cost, states)
    new_x = states - lr * grads
    if x_prior != 0:
        new_x += lambdav * lr * .1 * _proxInnov(states, x_prior)
    new_states = _proxOp(new_x, lr * lambdav)
    return theano.gradient.disconnected_grad(new_states)


def _RMSPropStep(cost, states, accum_1, accum_2):
    rho = .9
    lr = .001
    momentum = .9
    epsilon = 1e-8

    grads = T.grad(cost, states)

    new_accum_1 = rho * accum_1 + (1 - rho) * grads ** 2  # running average
    new_accum_2 = momentum * accum_2 - lr * grads / T.sqrt(new_accum_1 + epsilon)  # momentum

    new_states = states + momentum * new_accum_2 - lr * (grads /
                                                         T.sqrt(new_accum_1 + epsilon))
    return new_states, new_accum_1, new_accum_2


class SparseCoding(Layer):
    def __init__(self, input_dim,
                 output_dim,
                 n_steps=25,
                 gamma=.1,
                 init='glorot_uniform',
                 activation='linear',
                 batch_size=100,
                 return_reconstruction=True,  # We will return the W*A which approximates the input
                 truncate_gradient=1,
                 W_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.truncate_gradient = truncate_gradient
        self.return_reconstruction = return_reconstruction

        # Dictionary (k by d), Codes (n by k)
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

    def updating_codes(self,
                       x_t,
                       accum_1,
                       accum_2,
                       inputs
                       ):
        outputs = T.dot(x_t, self.W)

        rec_error = T.sqr(inputs - outputs).sum()  # First right hand term of L (s - Ax)^2
        l1_norm = T.sum(self.gamma * T.sqrt(T.power(x_t, 2) + 1e-7))

        cost = rec_error + l1_norm
        x, new_accum_1, new_accum_2 = _RMSPropStep(cost, x_t, accum_1, accum_2)
        # _RMSPropStep does the step
        # optimization x <- x + n*grad(L,x)
        return x, new_accum_1, new_accum_2, outputs

    def updating_dictionary(self, w, outputs):

        return w, outputs

    def call(self, inputs, mask=None):
        initial_codes = T.dmatrix("initial_codes")
        print "initial_codes.shape", initial_codes.type
        print "inputs.shape", inputs

        codes_outputs, _ = theano.scan(
            self.updating_codes,
            sequences=None,
            outputs_info=[initial_codes, ] * 3 + [None, ],  # Initilization
            non_sequences=[inputs, ],
            n_steps=self.n_steps)

        # dictionary_outputs, _ = theano.scan(
        #     self.updating_dictionary,
        #     sequences=None,
        #     outputs_info=[self.W, ],  # Initilization
        #     non_sequences=[inputs, ],
        #     n_steps=self.n_steps)

        print "outputs.shape", codes_outputs.shape
        final_output = codes_outputs[-1]

        if self.return_reconstruction:
            return T.dot(final_output, self.W)
        else:
            return final_output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "batch_size": self.batch_size,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
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
            output_dim=1000,
            n_steps=25,
            truncate_gradient=1,
            W_regularizer=l2(.00005),
            return_reconstruction=True
        )
    )

    print model.get_config()

    model.compile(loss='mse', optimizer='rmsprop')  # RMSprop for Maximization as well

    nb_epoch = 10
    batch_size = 100
    model.fit(S,  # input
              S,  # and output are the same thing, since we are doing generative modeling.
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=2)

    print "model.get_weights().shape", model.get_weights()[0].shape
    # I = grid2d(A)
    # plt.imshow(I)


if __name__ == "__main__":
    main()
