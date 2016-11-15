# Tran, Nhat
# 1000-787-456
# 2016-11-15
# Assignment_05

import theano
from theano import tensor as T
import numpy as np


def floatX(X):
    """
    Used to convert np arrays into theano matrices
    :param X:
    :return:
    """
    return np.asarray(X, dtype=theano.config.floatX)


class NeuralNet:
    def __init__(self, n_inputs=768, n_classes=10, n_hidden_nodes=100, alpha=0.1, n_epoch=200, activation='sigmoid'):
        """
        A neural network implementation using Theano for a one-hidden layer and output layer with 10 nodes

        :param n_hidden_nodes: Number of nodes in the hidden layer
        :param alpha: the coefficient for L-2 weight regularization
        :param n_epoch: Number of training epochs for SGD. Default: 200
        :param activation: Choice of activation method among ['sigmoid', 'softmax', 'relu']. Default: 'sigmoid'
        :param n_inputs: number of inputs (hard coded for assignment)
        :param n_classes: number of output nodes (hard coded for assignment)
        """
        self.activation = activation
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.n_hidden_nodes = n_hidden_nodes
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        self.layers = [theano.shared(floatX(np.random.rand((self.n_inputs, self.n_hidden_nodes)))),
                       theano.shared(floatX(np.random.rand((self.n_hidden_nodes, self.n_classes))))]

    def fprop(self, X):
        pass

    def loss(self, X):
        if self.activation == 'sigmoid':
            return T.nnet.sigmoid(X)
        elif self.activation == 'softmax':
            return T.nnet.softmax(X)
        elif self.activation == 'relu':
            return T.nnet.relu(X)
        else:
            return X

    def bprop(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


if __name__ == "__main__":
    pass
