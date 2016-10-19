import Tkinter as Tk
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from experiment import default_experiment


class MomentMatchingGui(object):
    def __init__(self):
        self.master = Tk.Tk()
        self.master.geometry('800x400')
        self.master.title('Moment Matching')
        self.frame = Tk.Frame(self.master)
        self.label = Tk.Label(self.frame,
                              text="\nOptimize toy data to match the target distribution by updating the values in the actual distribution to minimize MMD\n")
        self.label.pack()

        self.frame.pack(side=Tk.TOP, fill=Tk.X)

        self.experiment = self.initialize_experiment()
        self.create_plots()
        self.config_frame = Tk.Frame(self.master)
        self.train_button = Tk.Button(self.config_frame, text="Train", command=self.train)
        self.learning_rate = Tk.DoubleVar()
        self.batch_size = Tk.IntVar()

        self.learning_rate.set(10.0)
        self.batch_size.set(1024)

        self.learning_rate_slider = Tk.Scale(self.config_frame, from_=0, to=10.0, resolution=0.1, orient=Tk.HORIZONTAL,
                                             command=self.update_learning_rate, label="alpha",
                                             variable=self.learning_rate)
        self.batch_size_slider = Tk.Scale(self.config_frame, from_=2, to=5000, resolution=2, orient=Tk.HORIZONTAL,
                                          command=self.update_batch_size, label="Batch Size", variable=self.batch_size)
        self.config_frame.pack(side=Tk.TOP, fill=Tk.Y)
        self.train_button.pack(side=Tk.LEFT)
        self.batch_size_slider.pack(side=Tk.LEFT)
        self.learning_rate_slider.pack(side=Tk.LEFT)

    def update_learning_rate(self, val):
        self.experiment.alpha = self.learning_rate.get()

    def update_batch_size(self, val):
        self.experiment.batch_size = int(self.batch_size.get())

    def update_distribution(self):
        self.experiment.update()
        self.actual_distribution_plot.cla()
        self.actual_distribution_plot.set_title('Actual Distribution')
        self.actual_distribution_plot.hist(self.experiment.actual.squeeze(), bins=100, normed=1, facecolor='blue',
                                           alpha=1.0)
        self.canvas.draw()

    def create_plots(self):
        self.outer = plt.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        self.f = plt.figure(figsize=(8, 3))
        self.f.patch.set_alpha(0.0)

        self.target_distribution_plot = plt.subplot(self.outer[0, 0])
        self.target_distribution_plot.set_title('Target Distribution')
        self.actual_distribution_plot = plt.subplot(self.outer[0, 1])
        self.actual_distribution_plot.set_title('Actual Distribution')

        val, self.bins, patches = self.actual_distribution_plot.hist(self.experiment.actual.squeeze(), bins=100,
                                                                     normed=1, facecolor='blue', alpha=1.0)

        self.target_distribution_plot.hist(self.experiment.target.squeeze(), bins=self.bins, normed=1, facecolor='red',
                                           alpha=1.00)

        self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack(fill=Tk.BOTH)
        self.canvas.draw_idle()

    def initialize_experiment(self):
        return default_experiment()

    def train_loop(self):
        if self.running:
            self.update_distribution()
            self.master.after(25, self.train_loop)

    def train(self):
        self.running = True
        self.train_button.configure(text="Stop Training")
        self.train_button.configure(command=self.stop)
        self.train_loop()

    def stop(self):
        self.running = False
        self.train_button.configure(text="Train")
        self.train_button.configure(command=self.train)

    def run(self):
        self.master.mainloop()


if __name__ == "__main__":
    gui = MomentMatchingGui()
    gui.run()
