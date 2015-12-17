"""
author: Nhat Tran

"""

import numpy as np

class NeuralNetwork:

    def __init__(self,
                 lr=0.001,
                 dc=1e-12,
                 sizes=[20, 15, 5],
                 L2=0.001,
                 L1=0,
                 seed=1234,
                 n_epochs=100):
        """

        :param lr:
        :param dc:
        :param sizes:
        :param L2:
        :param L1:
        :param seed:
        :param n_epochs:
        """
        self.lr = lr
        self.lr_bk = lr
        self.dc = dc
        self.sizes = sizes
        self.L2 = L2
        self.L1 = L1
        self.seed = seed
        self.n_epochs = n_epochs

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
        # Initialize hidden layer weights & biases

        self.rng = np.random.mtrand.RandomState(self.seed)  # create random number generator
        # biases are initialized to zero
        # ... and weights according to the slides
        for m in range(len(self.weights)):
            b = (6 ** 0.5) / ((self.weights[m].shape[0] + self.weights[m].shape[1]) ** 0.5)
            for ind, val in np.ndenumerate(self.weights[m]):
                self.weights[m][ind] = self.rng.uniform(-b, b, 1)

        self.n_updates = 0  # To keep track of the number of updates, to decrease the learning rate

    def train(self, X, Y):

        for it in range(self.epoch, self.n_epochs):
            total_loss = 0
            random_range = range(X.shape[0])
            np.random.shuffle(random_range)

            for X_i in random_range:
                input = X[X_i]
                target = Y[X_i]

                if self.classes_mapping:
                    target = self.classes_mapping.index(target)

                total_loss += self.fprop(input, target)
                self.bprop(input, target)
                self.update()

            print "epoch:", self.epoch, ", loss:", total_loss / X.shape[0], "lr", self.lr

            self.epoch += 1


    def fprop(self, input, target):
        """

        :param input:
        :param target: The class (in range of 0 and n_classes-1)
        :return:
        """

        self.activation[0] = self.sigmoid_activation(input.dot(self.weights[0]) + self.biases[0])
        for it in range(1, len(self.weights) - 1):
            self.activation[it] = 1 / (1 + np.e ** -(self.activation[it - 1].dot(self.weights[it]) + self.biases[it]))

        # output layer: no act func applied
        self.activation[-1] = self.activation[-2].dot(self.weights[-1]) + self.biases[-1]

        # apply softmax
        self.activation[-1] = (np.e ** self.activation[-1]) / (np.e ** self.activation[-1]).sum(axis=0)

        return self.training_loss(self.activation[-1], target)

    def bprop(self, input, target):
        """

        :param input:
        :param target: The class (in range of 0 and n_classes-1)
        """

        e_y = np.zeros(self.n_classes)
        e_y[target] = 1
        self.activation_grad[-1] = -(e_y - self.activation[-1])

        for k in range(len(self.sizes), -1, -1):
            # compute grads of hidden layer params
            if k == 0:
                self.weights_grad[k] = np.outer(input, self.activation_grad[k])
            else:
                self.weights_grad[k] = np.outer(self.activation[k - 1], self.activation_grad[k])
            self.biases_grad[k] = self.activation_grad[k]

            if k > 0:
                grad_wrt_h = self.weights[k].dot(self.activation_grad[k])
                self.activation_grad[k - 1] = np.multiply(grad_wrt_h,
                                                          self.activation[k - 1] - self.activation[k - 1] ** 2)

        # add regularization gradients
        if self.L1 != 0:
            for k in range(0, len(self.weights)):
                self.weights_grad[k] += self.L1 * np.sign(self.weights[k])
        elif self.L2 != 0:
            for k in range(0, len(self.weights)):
                self.weights_grad[k] += self.L2 * 2 * self.weights[k]


    def update(self):
        # self.lr -= self.epoch * self.dc
        # if self.lr < 0:
        #     print("learning rate resetted")
        #     self.lr = self.lr_bk

        for h in range(len(self.weights)):
            self.weights[h] -= self.lr * self.weights_grad[h]
            self.biases[h] -= self.lr * self.biases_grad[h]

    def training_loss(self, output, target):
        """

        :param output:
        :param target:
        :return:
        """
        loss = -np.log(output[target])

        # Regularization copied from Larochelle
        if self.L1 != 0:
            for k in range(len(self.weights)):
                loss += self.L1 * abs(self.weights[k]).sum(axis=1).sum(axis=0)
        elif self.L2 != 0:
            for k in range(len(self.weights)):
                loss += self.L2 * (self.weights[k] ** 2).sum(axis=1).sum(axis=0)
        return loss

    @staticmethod
    def sigmoid_activation(preactivation):
        return 1 / (1 + np.power(np.e, -(preactivation)))

    @staticmethod
    def softmax_activation(preactivation):
        return np.power(np.e, preactivation) / np.power(np.e, preactivation).sum(axis=0)

    def predict(self, dataset):
        predictions = [0, ] * (dataset.shape[0])
        probability = [0, ] * (dataset.shape[0])

        for i, row in enumerate(dataset):
            self.fprop(row, 0)

            predictions[i] = np.argmax(self.activation[-1])
            probability[i] = self.activation[-1][predictions[i]]

            # Convert predicted class index to the class label
            predictions[i] = self.classes_mapping[int(predictions[i])]

        return predictions, probability

if __name__ == '__main__':
    m = NeuralNetwork()
    m.initialize(240, 4, classes_mapping=[4, 5, 6, 7])
    Y = np.random.randint(4) + 4
    for i in range(39):
        Y = np.vstack((Y, np.random.randint(4) + 4))
    X = np.random.rand(40, 240)
    print X.shape, Y.shape
    m.train(X, Y)

    print m.predict(np.random.rand(10, 240))
