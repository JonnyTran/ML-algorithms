"""
author: Nhat Tran

Copied code structure from Larochelle's online class (i.e. function names & parameters).
"""

import numpy as np

class NeuralNetwork:
    """
    Neural network for classification.

    Option ``lr`` is the learning rate.

    Option ``dc`` is the decrease constante for the learning rate.

    Option ``sizes`` is the list of hidden layer sizes.

    Option ``L2`` is the L2 regularization weight (weight decay).

    Option ``L1`` is the L1 regularization weight (weight decay).

    Option ``seed`` is the seed of the random number generator.

    Option ``tanh`` is a boolean indicating whether to use the
    hyperbolic tangent activation function (True) instead of the
    sigmoid activation function (True).

    Option ``n_epochs`` number of training epochs.

    **Required metadata:**

    * ``'input_size'``: Size of the input.csv.
    * ``'targets'``: Set of possible targets.

    """

    def __init__(self,
                 lr=0.001,
                 dc=1e-10,
                 sizes=[15, 10, 5],
                 L2=0.001,
                 L1=0,
                 seed=1234,
                 tanh=True,
                 n_epochs=10):
        self.lr = lr
        self.dc = dc
        self.sizes = sizes
        self.L2 = L2
        self.L1 = L1
        self.seed = seed
        self.tanh = tanh
        self.n_epochs = n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0

    def initialize(self, input_size, n_classes, classes_mapping=None):
        """

        :param input_size:
        :param n_classes:
        :param classes_mapping: An array of original class names of the raw data
        """
        self.n_classes = n_classes
        self.input_size = input_size
        self.classes_mapping = classes_mapping

        ###############################################################################
        # Allocate hidden layers' & output's activation

        self.activation = []
        self.activation_grad = []

        for h_size in self.sizes:
            self.activation += [np.zeros((h_size,))]
            self.activation_grad += [np.zeros((h_size,))]

        self.activation += [np.zeros((self.n_classes,))]
        self.activation_grad += [np.zeros((self.n_classes,))]

        ###############################################################################
        # Allocate hidden layer weights & biases

        # Input layer
        self.weights = [np.zeros((self.input_size, self.sizes[0]))]
        self.weights_grad = [np.zeros((self.input_size, self.sizes[0]))]

        self.biases = [np.zeros((self.sizes[0],))]
        self.biases_grad = [np.zeros((self.sizes[0],))]

        # Hidden layers
        for h in range(len(self.sizes) - 1):
            self.weights += [np.zeros((self.sizes[h], self.sizes[h + 1]))]
            self.weights_grad += [np.zeros((self.sizes[h], self.sizes[h + 1]))]

            self.biases += [np.zeros((self.sizes[h + 1],))]
            self.biases_grad += [np.zeros((self.sizes[h + 1],))]

        # Output layer
        self.weights += [np.zeros((self.sizes[-1], self.n_classes))]
        self.weights_grad += [np.zeros((self.sizes[-1], self.n_classes))]

        self.biases += [np.zeros((self.n_classes))]
        self.biases_grad += [np.zeros((self.n_classes))]

        ###############################################################################
        # Initialize hidden layer weights & biases (Copied from Larochelle)

        self.rng = np.random.mtrand.RandomState(self.seed)  # create random number generator
        # biases are initialized to zero
        # ... and weights according to the slides
        for m in range(len(self.weights)):
            b = (6 ** 0.5) / ((self.weights[m].shape[0] + self.weights[m].shape[1]) ** 0.5)
            for ind, val in np.ndenumerate(self.weights[m]):
                self.weights[m][ind] = self.rng.uniform(-b, b, 1)

        self.n_updates = 0  # To keep track of the number of updates, to decrease the learning rate

    def train(self, trainset):
        # if self.classes_mapping:
        # target = self.classes_mapping.index(target)
        pass


    def fprop(self, input, target):
        """

        :param input:
        :param target:
        :return:
        """

        # First layer activation
        self.activation[0] = self.sigmoid_activation(input.dot(self.weights[0]) + self.biases[0])

        # Hidden layer activations
        for h in range(1, len(self.sizes) - 1):
            self.activation[h + 1] = self.sigmoid_activation(
                self.activation[h].dot(self.weights[h + 1]) + self.biases[h + 1])

        # Output layer activations
        self.activation[-1] = self.softmax_activation(
            self.activation[-2].dot(self.weights[-1]) + self.biases[-1])

        return self.training_loss(self.hs[-1], target)

    def bprop(self, input, target):
        pass


    def update(self):
        for h in range(len(self.weights)):
            self.weights[h] -= self.lr * self.weights_grad[h]
            self.biases[h] -= self.lr * self.biases_grad[h]

    def training_loss(self, output, target):
        """
        Copied regularization from Larochelle

        :param output:
        :param target:
        :return:
        """
        loss = -np.log(output[target])

        # Regularization
        if self.L1 != 0:
            for k in range(len(self.weights)):
                loss += self.L1 * abs(self.weights[k]).sum(axis=1).sum(axis=0)
        elif self.L2 != 0:
            for k in range(len(self.weights)):
                loss += self.L2 * (self.weights[k] ** 2).sum(axis=1).sum(axis=0)
        return loss

    @staticmethod
    def sigmoid_activation(preactivation):
        return 1 / (1 + np.e ** -(preactivation))

    @staticmethod
    def softmax_activation(preactivation):
        return (np.e ** preactivation) / (np.e ** preactivation).sum(axis=0)

    def use(self, dataset):
        pass

    def test(self, testset):
        pass

def main():
    m = NeuralNetwork()
    m.initialize(20, 2)

if __name__ == '__main__':
    main()