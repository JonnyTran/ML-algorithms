"""
author: Nhat Tran

"""

import os
import struct
from array import array as pyarray

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import ion
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, int8, uint8, zeros
from sklearn.decomposition import sparse_encode, MiniBatchDictionaryLearning
from sklearn.svm import LinearSVC


def plot_gallery(title, images, titles, h, w, channel=1, n_row=10, n_col=10):
    plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))
    plt.title(title)
    # plt.subplots_adjust(bottom=0.1, left=.01, right=.99, top=.90, hspace=.1)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((w, h)))
        plt.title(titles[i], size=8)
        plt.xticks(())
        plt.yticks(())


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


###############################################################################
# Load the data


# Concatenate all 5 batches of data
train_X, train_Y = load_mnist('training')
train_X = -train_X.reshape((60000, 28 * 28))

test_X, test_Y = load_mnist('testing')
test_X = -test_X.reshape((10000, 28 * 28))

print('train_X.shape', train_X.shape)
print('train_Y.shape', train_Y.shape)
# print('X_test.shape', test_X.shape)

w, h = 28, 28
n_features = 28 * 28
n_samples = len(train_X)

print("Total dataset size:")
print("train n_samples: %d" % n_samples)
print("total n_features: %d" % n_features)

###############################################################################
# Dictionary Learning
n_components = 50

print("\nSparse Coding Dictionary Learning")
# pca = RandomizedPCA(n_components=n_components).fit(train_X)
dl = MiniBatchDictionaryLearning(n_components)
dl.fit(train_X)

print "X_train.shape", train_X.shape
print "Components shape", dl.components_.shape

# components = dl.components().reshape((n_components, n_features))
components = dl.components_

# Visualizing the components as images
component_titles = ["component %d" % i for i in range(components.shape[0])]
ion()
plot_gallery("Visualizing top components", components, component_titles, w, h, n_row=n_components / 10, n_col=10)
plt.show()

###############################################################################
# Sparse Encoding
print("\nSparse Encoding")
train_X_sc = np.zeros((10, n_components))
train_X_sc = sparse_encode(train_X, components, algorithm='omp')
np.set_printoptions(precision=1, suppress=False, linewidth=800)

test_X_sc = np.zeros((len(test_X), n_components))
test_X_sc = sparse_encode(test_X, components, algorithm='omp')

print "train_X_sc.shape", train_X_sc.shape

###############################################################################
# Visualize reconstructed images
reconstructed_X = np.zeros((20, n_features))
reconstructed_X[0:10] = train_X[0:10]
reconstructed_X[10:20] = np.dot(train_X_sc[0:10], components)

print "reconstructed_X.shape", reconstructed_X.shape
print(reconstructed_X)

reconstructed_titles = ["reconstructed %d" % i for i in range(20)]
ion()
plot_gallery("Reconstructed images", reconstructed_X, reconstructed_titles, w, h, n_row=2, n_col=10)
plt.show()

###############################################################################
# Train a neuralnet classification model

print("\nFitting the classifier to the training set")

# nnet = NeuralNetwork(lr=0.1, sizes=[50, 25, 10], seed=1234, n_epochs=50)
# nnet.initialize(n_components, 10, classes_mapping=np.arange(10))
clf = LinearSVC()
clf.fit(train_X_sc, train_Y)
test_y_pred = clf.predict(test_X_sc)

print test_y_pred
print "test_y_pred.shape", test_y_pred.shape

# i_iteration, i_loss, lr = nnet.train(train_X_sc, train_X_labels)

###############################################################################
# Quantitative evaluation of the prediction accuracy on the test set

print("\nPredicting classes on the test set")
# train_y_pred, train_y_prob = nnet.predict(train_X_sc)
# test_y_pred, test_y_prob = nnet.predict(test_X_pca)

# accuracy = 0.0
# for i in range(len(train_y)):
#     if train_y[i] == train_y_pred[i]:
#         accuracy += 1
# accuracy /= len(train_y)

# print("Classification accuracy = %f, among %d classes IN TRAINSET" % (accuracy, len(animal_labels)))

accuracy = 0.0
for i in range(len(test_Y)):
    if test_Y[i] == test_y_pred[i]:
        accuracy += 1
accuracy /= len(test_Y)

print("Classification accuracy = %f, among %d classes IN TESTSET" % (accuracy, len(animal_labels)))

###############################################################################
# Some visualization of learning progress, data point scatterplot

# Visualize neural net learning
plt.figure()
# plt.title("NNet Learning (Error vs. Epoch). Rate: %0.5f, Accuracy: %0.3f" % (lr, accuracy))
plt.xlabel("# of epoch iteration")
plt.ylabel("Training loss (negative log)")
# plt.plot(i_iteration, i_loss, marker='.')

# Visualize data points on top principle components on scatterplot
scatterplot = plt.figure()
ax = Axes3D(scatterplot)  # use the plotting figure to create a Axis3D object.
colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
# for i, animal in enumerate(animal_labels):
#     animal_indices = []
# for j in range(len(train_y)):
#     if train_y[j] == animal: animal_indices.append(j)

# ax.scatter(train_X_sc[animal_indices][0], train_X_sc[animal_indices][1], train_X_sc[animal_indices][2],
# color=colors[i % len(colors)], label=all_labels[animal])

ax.set_xlabel("1st principal component")
ax.set_ylabel("2nd principal component")
ax.set_zlabel("3rd principal component")
ax.set_title("Scatter plot of all images and their classes")
