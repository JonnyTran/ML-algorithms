import numpy as np


class NeuralNetwork():
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
                 sizes=[200, 100, 50],
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
        """
        This method allocates memory for the fprop/bprop computations (DONE)
        and initializes the parameters of the neural network (TODO)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        n_hidden_layers = len(self.sizes)
        #############################################################################
        # Allocate space for the hidden and output layers, as well as the gradients #
        #############################################################################
        self.hs = []
        self.grad_hs = []
        for h in range(n_hidden_layers):
            self.hs += [np.zeros((self.sizes[h],))]  # hidden layer
            self.grad_hs += [np.zeros((self.sizes[h],))]  # ... and gradient
        self.hs += [np.zeros((self.n_classes,))]  # output layer
        self.grad_hs += [np.zeros((self.n_classes,))]  # ... and gradient

        ##################################################################
        # Allocate space for the neural network parameters and gradients #
        ##################################################################
        self.weights = [np.zeros((self.input_size, self.sizes[0]))]  # input.csv to 1st hidden layer weights
        self.grad_weights = [np.zeros((self.input_size, self.sizes[0]))]  # ... and gradient

        self.biases = [np.zeros((self.sizes[0]))]  # 1st hidden layer biases
        self.grad_biases = [np.zeros((self.sizes[0]))]  # ... and gradient

        for h in range(1, n_hidden_layers):
            self.weights += [np.zeros((self.sizes[h - 1], self.sizes[h]))]  # h-1 to h hidden layer weights
            self.grad_weights += [np.zeros((self.sizes[h - 1], self.sizes[h]))]  # ... and gradient

            self.biases += [np.zeros((self.sizes[h]))]  # hth hidden layer biases
            self.grad_biases += [np.zeros((self.sizes[h]))]  # ... and gradient

        self.weights += [np.zeros((self.sizes[-1], self.n_classes))]  # last hidden to output layer weights
        self.grad_weights += [np.zeros((self.sizes[-1], self.n_classes))]  # ... and gradient

        self.biases += [np.zeros((self.n_classes))]  # output layer biases
        self.grad_biases += [np.zeros((self.n_classes))]  # ... and gradient

        #########################
        # Initialize parameters #
        #########################

        self.rng = np.random.mtrand.RandomState(self.seed)  # create random number generator
        # biases are initialized to zero
        # ... and weights according to the slides
        for m in range(len(self.weights)):
            b = (6 ** 0.5) / ((self.weights[m].shape[0] + self.weights[m].shape[1]) ** 0.5)
            for ind, val in np.ndenumerate(self.weights[m]):
                self.weights[m][ind] = self.rng.uniform(-b, b, 1)


        self.n_updates = 0  # To keep track of the number of updates, to decrease the learning rate

    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size, self.targets)
        self.epoch = 0

    def train(self, trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until 
        ``self.epoch == self.n_epochs``.
        
        If ``self.epoch == 0``, first initialize the model.
        """

        if self.epoch == 0:
            input_size = trainset.metadata['input_size']
            n_classes = len(trainset.metadata['targets'])
            self.initialize(input_size, n_classes)

        for it in range(self.epoch, self.n_epochs):
            for input, target in trainset:
                self.fprop(input, target)
                self.bprop(input, target)
                self.update()
        self.epoch = self.n_epochs

    def fprop(self, input, target):
        """
        Forward propagation: 
        - fills the hidden layers and output layer in self.hs
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input.csv``,``target``) pair
        Argument ``input.csv`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        self.hs[0] = 1 / (1 + np.e ** -(input.dot(self.weights[0]) + self.biases[0]))
        for it in range(1, len(self.weights) - 1):
            self.hs[it] = 1 / (1 + np.e ** -(self.hs[it - 1].dot(self.weights[it]) + self.biases[it]))

        # output layer: no act func applied
        self.hs[-1] = self.hs[-2].dot(self.weights[-1]) + self.biases[-1]

        # apply softmax
        self.hs[-1] = (np.e ** self.hs[-1]) / (np.e ** self.hs[-1]).sum(axis=0)
        return self.training_loss(self.hs[-1], target)


    def training_loss(self, output, target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given output vector (probabilities of each class) and target (class ID)
        """

        loss = -np.log(output[target])

        # add regularization
        if self.L1 != 0:
            for k in range(len(self.weights)):
                loss += self.L1 * abs(self.weights[k]).sum(axis=1).sum(axis=0)
        elif self.L2 != 0:
            for k in range(len(self.weights)):
                loss += self.L2 * (self.weights[k] ** 2).sum(axis=1).sum(axis=0)
        return loss

    def bprop(self, input, target):
        """
        Backpropagation:
        - fills in the hidden layers and output layer gradients in self.grad_hs
        - fills in the neural network gradients of weights and biases in self.grad_weights and self.grad_biases
        - returns nothing
        Argument ``input.csv`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        # step 1: computing output gradients (before activation) depends on output activation function, obviously :-)
        # ... here is softmax for the last layer
        e_y = np.zeros(self.n_classes)
        e_y[target] = 1
        self.grad_hs[-1] = -(e_y - self.hs[-1])

        # step 2: backpropagate grads through hidden layers
        for k in range(len(self.sizes), -1, -1):
            # compute grads of hidden layer params
            if k == 0:
                self.grad_weights[k] = np.outer(input, self.grad_hs[k])
            else:
                self.grad_weights[k] = np.outer(self.hs[k - 1], self.grad_hs[k])
            self.grad_biases[k] = self.grad_hs[k]

            if k > 0:  # hidden layer below exists!
                print "hidden layer below exists!", self.epoch
                # compute grads of hidden layer below
                grad_wrt_h = self.weights[k].dot(self.grad_hs[k])
                # compute grads of hidden layer below (before activation)
                if self.tanh == True:
                    self.grad_hs[k - 1] = np.multiply(grad_wrt_h, 1 - self.hs[k - 1] ** 2)
                else:
                    self.grad_hs[k - 1] = np.multiply(grad_wrt_h, self.hs[k - 1] - self.hs[k - 1] ** 2)

        # add regularization gradients
        if self.L1 != 0:
            for k in range(0, len(self.weights)):
                self.grad_weights[k] += self.L1 * np.sign(self.weights[k])
        elif self.L2 != 0:
            for k in range(0, len(self.weights)):
                self.grad_weights[k] += self.L2 * 2 * self.weights[k]

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the neural network parameters self.weights and self.biases,
          using the gradients in self.grad_weights and self.grad_biases
        """

        for k in range(len(self.weights)):
            self.weights[k] -= self.lr * self.grad_weights[k]
            self.biases[k] -= self.lr * self.grad_biases[k]

    def use(self, dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a Numpy 2D array of size
          len(dataset) by (nb of classes + 1)
        - the ith row of the array contains the outputs for the ith example
        - the outputs for each example should contain
          the predicted class (first element) and the
          output probabilities for each class (following elements)
        Argument ``dataset`` is an MLProblem object.
        """

        outputs = np.zeros((len(dataset), self.n_classes + 1))
        errors = np.zeros((len(dataset), 2))

        ## PUT CODE HERE ##
        # row[0] is input.csv image (array), row[1] actual target class for that image
        for ind, row in enumerate(dataset):
            # fill 2nd element with loss
            errors[ind, 1] = self.fprop(row[0], row[1])
            # predicted class
            outputs[ind, 0] = np.argmax(self.hs[-1])
            # 0/1 classification error
            errors[ind, 0] = (outputs[ind, 0] != row[1])
            #             print "errors: ", errors[ind, ]
            # add output probs
            np.copyto(outputs[ind, 1:], self.hs[-1])
        #             print "outputs: ", outputs[ind,]
        #             time.sleep(5)

        return outputs, errors

    def test(self, dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of 
        those outputs for ``dataset``:
        - the errors should be a Numpy 2D array of size
          len(dataset) by 2
        - the ith row of the array contains the errors for the ith example
        - the errors for each example should contain 
          the 0/1 classification error (first element) and the 
          regularized negative log-likelihood (second element)
        Argument ``dataset`` is an MLProblem object.
        """

        outputs, errors = self.use(dataset)

        ## PUT CODE HERE ##
        # I put the code in the "use" function, seems better :-)

        return outputs, errors

    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        """

        print 'WARNING: calling verify_gradients reinitializes the learner'

        rng = np.random.mtrand.RandomState(1234)

        self.seed = 1234
        self.sizes = [4, 5]
        self.initialize(20, 3)
        example = (rng.rand(20) < 0.5, 2)
        input, target = example
        epsilon = 1e-6
        self.lr = 0.1
        self.decrease_constant = 0

        self.fprop(input, target)
        self.bprop(input, target)  # compute gradients

        import copy
        emp_grad_weights = copy.deepcopy(self.weights)

        for h in range(len(self.weights)):
            for i in range(self.weights[h].shape[0]):
                for j in range(self.weights[h].shape[1]):
                    self.weights[h][i, j] += epsilon
                    a = self.fprop(input, target)
                    self.weights[h][i, j] -= epsilon

                    self.weights[h][i, j] -= epsilon
                    b = self.fprop(input, target)
                    self.weights[h][i, j] += epsilon

                    emp_grad_weights[h][i, j] = (a - b) / (2. * epsilon)

        print 'grad_weights[0] diff.:', np.sum(np.abs(self.grad_weights[0].ravel() - emp_grad_weights[0].ravel())) / \
                                        self.weights[0].ravel().shape[0]
        print 'grad_weights[1] diff.:', np.sum(np.abs(self.grad_weights[1].ravel() - emp_grad_weights[1].ravel())) / \
                                        self.weights[1].ravel().shape[0]
        print 'grad_weights[2] diff.:', np.sum(np.abs(self.grad_weights[2].ravel() - emp_grad_weights[2].ravel())) / \
                                        self.weights[2].ravel().shape[0]

        emp_grad_biases = copy.deepcopy(self.biases)
        for h in range(len(self.biases)):
            for i in range(self.biases[h].shape[0]):
                self.biases[h][i] += epsilon
                a = self.fprop(input, target)
                self.biases[h][i] -= epsilon

                self.biases[h][i] -= epsilon
                b = self.fprop(input, target)
                self.biases[h][i] += epsilon

                emp_grad_biases[h][i] = (a - b) / (2. * epsilon)

        print 'grad_biases[0] diff.:', np.sum(np.abs(self.grad_biases[0].ravel() - emp_grad_biases[0].ravel())) / \
                                       self.biases[0].ravel().shape[0]
        print 'grad_biases[1] diff.:', np.sum(np.abs(self.grad_biases[1].ravel() - emp_grad_biases[1].ravel())) / \
                                       self.biases[1].ravel().shape[0]
        print 'grad_biases[2] diff.:', np.sum(np.abs(self.grad_biases[2].ravel() - emp_grad_biases[2].ravel())) / \
                                       self.biases[2].ravel().shape[0]


def main():
    m = NeuralNetwork()
    m.initialize(240, 2)


if __name__ == "__main__":
    main()
