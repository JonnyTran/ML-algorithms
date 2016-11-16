# Tran, Nhat
# 1000-787-456
# 2016-11-15
# Assignment_05

import numpy as np
import theano
from theano import tensor as T
from random import shuffle

def floatX(X):
    """
    Used to convert np arrays into theano matrices
    """
    return np.asarray(X, dtype=theano.config.floatX)


class NeuralNet:
    def __init__(self, n_inputs=1024, n_classes=10, n_hidden_nodes=100, alpha=0.1, lr=0.05, n_epoch=200,
                 activation='sigmoid'):
        """
        A neural network implementation using Theano for a one-hidden layer and output layer with 10 nodes

        :param n_hidden_nodes:
            Number of nodes in the hidden layer
        :param alpha:
            the coefficient for L-2 weight regularization
        :param n_epoch:
            Number of training epochs for SGD. Default: 200
        :param activation:
            Choice of activation method among ['sigmoid', 'relu', 'linear']. Default: 'sigmoid'
        :param n_inputs:
            number of inputs (hard coded for assignment)
        :param n_classes:
            number of output nodes (hard coded for assignment)
        """
        self.activation = activation
        self.n_epoch = n_epoch
        self.n_hidden_nodes = n_hidden_nodes
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        # Initialize Weights & Theano variables & symbolic equations
        X = T.matrix('X')
        y = T.matrix('y')

        self.layers = [
            theano.shared(name="W_hidden", value=floatX(np.random.rand(self.n_inputs, self.n_hidden_nodes) - 0.5)),
            theano.shared(name="W_output", value=floatX(np.random.rand(self.n_hidden_nodes, self.n_classes) - 0.5))]

        self.lr = theano.shared(floatX(lr))
        self.alpha = theano.shared(floatX(alpha))

        if self.activation == 'sigmoid':
            self.fprop = T.dot(T.nnet.sigmoid(T.dot(X, self.layers[0])), self.layers[1])
        elif self.activation == 'relu':
            self.fprop = T.dot(T.nnet.relu(T.dot(X, self.layers[0])), self.layers[1])
        else:
            self.fprop = T.dot(T.dot(X, self.layers[0]), self.layers[1])

        self.regularization = 0.5 * self.alpha * T.sum(T.power(self.layers[0], 2)) + \
                              0.5 * self.alpha * T.sum(T.power(self.layers[1], 2))  # TODO check L2 formula

        self.loss = T.mean((T.nnet.softmax(self.fprop) - y) ** 2) + self.regularization

        gradient_hidden = T.grad(cost=self.loss, wrt=self.layers[0])
        gradient_output = T.grad(cost=self.loss, wrt=self.layers[1])
        self.update = [(self.layers[0], self.layers[0] - gradient_hidden * self.lr),
                       (self.layers[1], self.layers[1] - gradient_output * self.lr)]

        self.fit = theano.function(inputs=[X, y], outputs=self.loss, updates=self.update, allow_input_downcast=True)

        self.predict_ = theano.function(inputs=[X], outputs=T.argmax(T.nnet.softmax(self.fprop), axis=1),
                                        allow_input_downcast=True)

    def train(self, X, y, batch_size):
        self.losses_ = []
        n_samples = X.shape[0]
        for i in range(self.n_epoch):
            batch = range(n_samples)
            shuffle(batch)

            self.losses_.append(self.fit(X[batch[0:batch_size]], y[batch[0:batch_size]]))

    def predict(self, X):
        return self.predict_(X)



if __name__ == "__main__":
    pass
    # trX = np.random.rand((100, 28*28))
    # trY = np.eye(100)[]
    #
    # train_X = train_X.reshape((60000, 28 * 28))
    # train_Y = train_Y.reshape((60000, 1))

    # new_train_Y = np.zeros((60000, 10))
    # for i in range(60000):
    #     new_train_Y[i] = np.eye(10)[train_Y[i]]
    #
    # nnet = NeuralNet(n_inputs=28 * 28, n_classes=10, n_hidden_nodes=100, alpha=0.1, n_epoch=200, activation='sigmoid')
    # nnet.train(train_X, new_train_Y)
    # print nnet.predict(train_X)
