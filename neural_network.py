import numpy as np
import os
from random import shuffle

class HiddenLayer:
    def __init__(self, hidden_unit_no, prev_unit_no):
        self.theta = np.random.rand(prev_unit_no+1, hidden_unit_no)
        self.input = []
        self.output = []
        self.error = []

    def activation(self, input, input_layer=False, output_layer=False):
        """
        sigmoid(theta' * activation value)

        :param input: a vector, the size of previous layer unit number
        :return: a vector, to be activation value for next layer
        """

        if (input_layer==True):
            input = np.array([1,] + input) # add 1 to input for bias multiplication
        else:
            input = np.array(input)

        product = np.dot(np.transpose(self.theta), input)
        self.input = input

        if (output_layer==True):
            self.output = [HiddenLayer.sigmoid(val) for val in product]
        else:
            self.output = [1,] + [HiddenLayer.sigmoid(val) for val in product]

        return self.output

    def error_update(self, succeeding_error, succeeding_theta):
        activation_deriv = [val*(1-val) for val in self.output]
        self.error = np.multiply(np.multiply(succeeding_theta.transpose(), succeeding_error), activation_deriv).transpose()

    def gradient_update(self):
        if (len(self.error) > 1):
            self.error = np.delete(self.error, 0)

        gradient = np.dot(np.matrix(self.input).transpose(), np.matrix(self.error))
        # self.theta = np.add(self.theta, gradient)

        return gradient

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def set_error(self, val):
        self.error = val
    def get_error(self):
        return self.error
    def get_theta(self):
        return self.theta
    def update_theta(self, gradient):
        self.theta = np.add(self.theta, gradient)


class NeuralNetwork:
    def __init__(self, layers, training_size):
        """

        :param layers: an array of number of units in each layer.
        E.g. [10, 4, 1] means 10 input units, 4 hidden units, and 1 output unit
        """
        self.layers = []
        for i in range(1, len(layers)):
            layer = HiddenLayer(prev_unit_no=layers[i-1], hidden_unit_no=layers[i])
            self.layers.append(layer)

        self.training_size = training_size

        self.deltas = [0,]*len(self.layers)

    def forward_prop(self, input):
        activation_val = self.layers[0].activation(input, input_layer=True)
        for i in range(1, len(self.layers)):
            if (i == len(self.layers)-1):
                activation_val = self.layers[i].activation(activation_val, output_layer=True)
            else:
                activation_val = self.layers[i].activation(activation_val)

        return activation_val

    def back_prop(self, output_error):
        # Calculate error deltas
        self.layers[len(self.layers)-1].set_error(output_error)
        for i in range(len(self.layers)-2, -1, -1): # Loop through the hidden layers before the output layer
            succeeding_error = self.layers[i+1].get_error()
            succeeding_theta = self.layers[i+1].get_theta()
            self.layers[i].error_update(succeeding_error, succeeding_theta)

        # Update gradients
        for i in range(len(self.layers)-1, -1, -1): # Loop through the hidden layers
            gradient = self.layers[i].gradient_update()
            if (self.deltas[i] == 0):
                self.deltas[i] = gradient
            else:
                self.deltas[i] = np.divide(np.add(self.deltas[i], gradient), self.training_size)

            self.layers[i].update_theta(self.deltas[i])

    def train(self, x, y):
        prediction = self.forward_prop(x)
        print "error:",np.power(np.subtract(prediction, y),2)
        self.back_prop(np.subtract(prediction, y))


def main():
    directory = os.getcwd()+"/images/Training/"
    training_files = os.listdir(directory)
    randomRange = range(len(training_files))

    myNN = NeuralNetwork(layers=[240, 4, 1], training_size=len(training_files))

    for i in range(200):
        shuffle(randomRange)
        for i in randomRange:
            if training_files[i].endswith(".pgm"):
                if training_files[i].startswith("ball"):
                    data = [line.strip() for line in open(directory+training_files[i], 'r')][3:243]
                    data = map(int, data)
                    print data
                    # myNN.train(x=[data], y=0)
                elif training_files[i].startswith("tree"):
                    data = [line.strip() for line in open(directory+training_files[i], 'r')][3:243]
                    data = map(int, data)
                    # myNN.train(x=[data], y=0)


if __name__ == "__main__":
    main()
