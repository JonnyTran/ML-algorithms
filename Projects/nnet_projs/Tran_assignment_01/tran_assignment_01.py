# Tran, Nhat
# 1000-787-456
# 2016-09-08
# Assignment_01

import Tkinter as Tk

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class CL_ActivationFunctions():
    """
    This class displays different activation functions
    Farhad Kamangar 2016_08_06
    """

    def __init__(self, master):
        self.master = master
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.width = 500
        self.height = 500
        self.xmin = -10
        self.xmax = 10
        self.ymin = -1
        self.ymax = 1
        self.input_weight = 1
        self.bias = 0
        self.learning_method = "Sigmoid"
        #########################################################################
        #  Set up the master frame
        #########################################################################
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")
        #########################################################################
        #  Set up the drawing canvas
        #########################################################################
        self.canvas = Tk.Canvas(self.master, height=self.height, width=self.width)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("Activation Functions")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='xx')
        # set up the sliders
        self.input_weight_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight",
                                            command=lambda event: self.input_weight_slider_callback())
        self.input_weight_slider.set(self.input_weight)
        self.input_weight_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback())
        self.input_weight_slider.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.bias_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias", length=800, width=20,
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.label_for_entry_box = Tk.Label(self.buttons_frame, text="Activation Function", justify="center")
        self.label_for_entry_box.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.learning_method_variable = Tk.StringVar()
        self.learning_method_dropdown = Tk.OptionMenu(self.buttons_frame, self.learning_method_variable, "Sigmoid",
                                                      "Linear", "Hyperbolic Tangent", "Positive Linear",
                                                      command=lambda event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("Sigmoid")
        self.learning_method_dropdown.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_activation_function()

    def display_activation_function(self):
        input_values = np.linspace(-10, 10, 256, endpoint=True)
        net_value = self.input_weight * input_values + self.bias
        if self.learning_method == 'Sigmoid':
            activation = 1.0 / (1 + np.exp(-net_value))
        elif self.learning_method == "Linear":
            activation = net_value
        elif self.learning_method == "Hyperbolic Tangent":
            activation = (np.power(np.e, net_value) - np.power(np.e, -net_value)) / (
            np.power(np.e, net_value) + np.power(np.e, -net_value))
        elif self.learning_method == "Positive Linear":
            activation = np.maximum(0, net_value)
        self.axes.cla()
        self.axes.plot(input_values, activation)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        plt.title(self.learning_method)
        # plt.title('Left Title', loc='left')
        # plt.title('Right Title', loc='right')
        self.canvas.draw()

    def input_weight_slider_callback(self):
        self.input_weight = self.input_weight_slider.get()
        self.display_activation_function()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        self.display_activation_function()

    def learning_method_dropdown_callback(self):
        self.learning_method = self.learning_method_variable.get()
        self.display_activation_function()


if __name__ == "__main__":
    main_frame = Tk.Tk()
    main_frame.title("Activation Functions")
    ob_activation_functions = CL_ActivationFunctions(main_frame)
    main_frame.mainloop()
