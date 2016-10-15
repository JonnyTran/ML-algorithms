# Tran, Nhat
# 1000-787-456
# 2016-10-16
# Assignment_04

import Tkinter as Tk

import os
import matplotlib
import numpy as np
from read_csv_data_and_convert_to_vector import read_csv_as_matrix

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from random import shuffle


class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        # The default values for samples and targets represent an exclusive or
        # Farhad Kamangar 2016_09_05
        self.samples = np.array(samples)

        # Get data
        directory = os.getcwd() + "/stock_data.csv"

        self.samples = read_csv_as_matrix(directory)

        self.targets = None

        print self.samples.shape
        # print self.targets.shape
        # if targets != None:
        #     self.targets = np.array(targets)
        # else:
        #     self.targets = None


nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 784,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_classes': 10,
    'number_of_samples_in_each_class': 100
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."

    def start_learning_epochs(self):
        print "self.data_set.samples.shape", self.data_set.samples.shape
        print "self.data_set.targets.shape", self.data_set.targets.shape

        errors = []
        for i in range(1000):
            randomRange = range(1000)
            shuffle(randomRange)
            error = self.neural_network.adjust_weights(self.data_set.samples[:, randomRange],
                                                       self.desired_target_vectors[:,
                                                       randomRange] - self.neural_network.calculate_output(
                                                           self.data_set.samples[:, randomRange]),
                                                       self.desired_target_vectors[:, randomRange],
                                                       self.learning_rate)
            errors.append(error)

        matplotlib.pyplot.close()
        matplotlib.pyplot.plot(errors)
        matplotlib.pyplot.show()


class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 3
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.learning_rate = self.nn_experiment.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 0.02
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("Hebbian Learning")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # self.number_of_classes_slider_label = Tk.Label(self.sliders_frame, text="Number of Classes")
        # self.number_of_classes_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_classes_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
        #                                          from_=2, to_=5, bg="#DDDDDD",
        #                                          activebackground="#FF0000",
        #                                          highlightcolor="#00FFFF", width=10)
        # self.number_of_classes_slider.set(self.number_of_classes)
        # self.number_of_classes_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_classes_slider_callback())
        # self.number_of_classes_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_samples_slider = Tk.Scale(self.sliders_frame, variable=ivar, orient=Tk.HORIZONTAL,
        #                                          from_=2, to_=20, bg="#DDDDDD",
        #                                          activebackground="#FF0000",
        #                                          highlightcolor="#00FFFF", width=10)
        # self.number_of_samples_slider_label = Tk.Label(self.sliders_frame, text="Number of Samples")
        # self.number_of_samples_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_samples_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_samples_slider_callback())
        # self.number_of_samples_slider.set(self.number_of_samples_in_each_class)
        # self.number_of_samples_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.create_new_samples_bottun = Tk.Button(self.buttons_frame,
        #                                            text="Create New Samples",
        #                                            bg="yellow", fg="red",
        #                                            command=lambda: self.create_new_samples_bottun_callback())
        # self.create_new_samples_bottun.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.randomize_weights_button = Tk.Button(self.buttons_frame,
        #                                           text="Randomize Weights",
        #                                           bg="yellow", fg="red",
        #                                           command=lambda: self.randomize_weights_button_callback())
        # self.randomize_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())

        self.learning_method_variable = Tk.StringVar()
        self.learning_method_dropdown = Tk.OptionMenu(self.buttons_frame, self.learning_method_variable,
                                                      "Filtered Learning",
                                                      "Delta Rule", "Unsupervised Hebb",
                                                      command=lambda event: self.learning_method_dropdown_callback())
        self.learning_method_dropdown.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_method_variable.set("Filtered Learning")

        self.print_nn_parameters_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.initialize()
        self.refresh_display()

    def learning_method_dropdown_callback(self):
        self.nn_experiment.neural_network.learning_method = self.learning_method_variable.get()
        print self.nn_experiment.neural_network.learning_method, "selected"

    def initialize(self):
        # self.nn_experiment.create_samples()
        # self.nn_experiment.desired_target_vectors = self.nn_experiment.dataset.targets
        self.nn_experiment.desired_target_vectors = self.nn_experiment.data_set.targets

        self.nn_experiment.neural_network.randomize_weights()
        self.neighborhood_colors = plt.cm.get_cmap('Accent')
        self.sample_points_colors = plt.cm.get_cmap('Dark2')
        self.xx, self.yy = np.meshgrid(np.arange(self.xmin, self.xmax + 0.5 * self.step_size, self.step_size),
                                       np.arange(self.ymin, self.ymax + 0.5 * self.step_size, self.step_size))
        self.convert_binary_to_integer = []
        for k in range(0, self.nn_experiment.neural_network.layers[-1].number_of_neurons):
            self.convert_binary_to_integer.append(2 ** k)

    def display_samples_on_image(self):
        # Display the samples for each class
        for class_index in range(0, self.number_of_classes):
            self.axes.scatter(self.nn_experiment.data_set.samples[0, class_index * self.number_of_samples_in_each_class: \
                (class_index + 1) * self.number_of_samples_in_each_class],
                              self.nn_experiment.data_set.samples[1, class_index * self.number_of_samples_in_each_class: \
                                  (class_index + 1) * self.number_of_samples_in_each_class],
                              c=self.sample_points_colors(class_index * (1.0 / self.number_of_classes)),
                              marker=(3 + class_index, 1, 0), s=50)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def refresh_display(self):
        pass
        # self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.samples)
        # self.display_neighborhoods()

    def display_neighborhoods(self):
        self.class_ids = []
        for x, y in np.stack((self.xx.ravel(), self.yy.ravel()), axis=-1):
            output = self.nn_experiment.neural_network.calculate_output(np.array([x, y]))
            self.class_ids.append(output.dot(self.convert_binary_to_integer))
        self.class_ids = np.array(self.class_ids)
        self.class_ids = self.class_ids.reshape(self.xx.shape)
        self.axes.cla()
        self.axes.pcolormesh(self.xx, self.yy, self.class_ids, cmap=self.neighborhood_colors)
        self.display_output_nodes_net_boundaries()
        self.display_samples_on_image()

    def display_output_nodes_net_boundaries(self):
        output_layer = self.nn_experiment.neural_network.layers[-1]
        for output_neuron_index in range(output_layer.number_of_neurons):
            w1 = output_layer.weights[output_neuron_index][0]
            w2 = output_layer.weights[output_neuron_index][1]
            w3 = output_layer.weights[output_neuron_index][2]
            if w1 == 0 and w2 == 0:
                data = [(0, 0), (0, 0), 'r']
            elif w1 == 0:
                data = [(self.xmin, self.xmax), (float(w3) / w2, float(w3) / w2), 'r']
            elif w2 == 0:
                data = [(float(-w3) / w1, float(-w3) / w1), (self.ymin, self.ymax), 'r']
            else:
                data = [(self.xmin, self.xmax),  # in form of (x1, x2), (y1, y2)
                        ((-w3 - float(w1 * self.xmin)) / w2,
                         (-w3 - float(w1 * self.xmax)) / w2), 'r']
            self.axes.plot(*data)

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.refresh_display()

    def number_of_classes_slider_callback(self):
        self.number_of_classes = self.number_of_classes_slider.get()
        self.nn_experiment.number_of_classes = self.number_of_classes
        self.nn_experiment.neural_network.layers[-1].number_of_neurons = self.number_of_classes
        self.nn_experiment.neural_network.randomize_weights()
        self.initialize()
        self.refresh_display()

    def number_of_samples_slider_callback(self):
        self.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.create_samples()
        self.refresh_display()

    def create_new_samples_bottun_callback(self):
        temp_text = self.create_new_samples_bottun.config('text')[-1]
        self.create_new_samples_bottun.config(text='Please Wait')
        self.create_new_samples_bottun.update_idletasks()
        # self.nn_experiment.create_samples()
        self.refresh_display()
        self.create_new_samples_bottun.config(text=temp_text)
        self.create_new_samples_bottun.update_idletasks()

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        for k in range(1):
            self.nn_experiment.start_learning_epochs()
            self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        # self.nn_experiment.neural_network.display_network_parameters()
        # self.nn_experiment.run_forward_pass()
        self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()


neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 784,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10,
                              "activation_function": "linear"}],  # list of dictionaries
    "learning_method": "Filtered Learning"
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights(self.min_initial_weights, self.max_initial_weights)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights.shape, \
                "\nLearning Rate : ", self.learning_rate

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)

        max_indices = np.argmax(output, axis=0)
        output = np.eye(10)[max_indices[0]].reshape((10, -1))
        for i in range(1, len(max_indices)):
            output = np.append(output, np.eye(10)[max_indices[i]].reshape((10, -1)), axis=1)

        self.output = output
        return self.output

    def adjust_weights(self, input_samples, error, targets, learning_rate):
        n_samples = input_samples.shape[1]
        input_matrix_with_ones = np.vstack((input_samples, np.ones(n_samples)))  # Adding ones to input matrix for bias

        mse = np.mean(np.linalg.norm(error, axis=0))
        print "Mean squared error:", mse

        if self.learning_method == "Filtered Learning":
            self.layers[0].weights = \
                self.layers[0].weights + learning_rate * np.dot(targets, input_matrix_with_ones.T)
        elif self.learning_method == "Delta Rule":
            self.layers[0].weights = \
                self.layers[0].weights + learning_rate * np.dot(error, input_matrix_with_ones.T)
        elif self.learning_method == "Unsupervised Hebb":
            self.layers[0].weights = \
                self.layers[0].weights + learning_rate * np.dot(self.output, input_matrix_with_ones.T)
        else:
            pass
        return mse


single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs_to_layer": 784,  # number of input signals
    "number_of_neurons": 10,  # number of neurons in the layer
    "activation_function": "linear"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
                                         (self.number_of_neurons, self.number_of_inputs_to_layer + 1))
        print "initialized weights", self.weights.shape

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        # Farhad Kamangar Sept. 4, 2016

        # print "ClSingleLayer.calculate_output"
        # print "input_values", input_values.shape
        if len(input_values.shape) == 1:
            net = self.weights.dot(np.append(input_values, 1))
        else:
            net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -0.1,  # minimum initial weight
        "max_initial_weights": 0.1,  # maximum initial weight
        "number_of_inputs": 784,  # number of inputs to the network
        "learning_rate": 0.1,  # learning rate
        "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
        "data_set": ClDataSet(),
        'number_of_classes': 10,
        'number_of_samples_in_each_class': 100
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Hebbian Learning")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()
