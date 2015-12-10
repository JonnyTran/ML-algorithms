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

    def initialize(self, input_size, n_classes):
        self.n_classes = n_classes
        self.input_size = input_size

        n_hidden_layers = len(self.sizes)
        self.hs = []

        for h_size in self.sizes:
            self.hs += [np.zeros((h_size,))]

        self.hs += [np.zeros((n_classes,))]



    def train(self, trainset):
        pass

    def fprop(self, input, target):
        pass

    def bprop(self, input, target):
        pass

    def update(self):
        pass

    def training_loss(self, output, target):
        pass

    def use(self, dataset):
        pass

    def test(self, testset):
        pass

def main():
    m = NeuralNetwork()
    m.initialize(20, 2)

if __name__ == '__main__':
    main()