"""
author: Nhat Tran

"""

import os
import struct
from array import array as pyarray

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, int8, uint8, zeros
from sklearn.svm import LinearSVC
from sparse_filtering import SparseFiltering

from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding


def plot_gallery(title, images, h, w, channel=1, n_row=10, n_col=10):
    # plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))
    plt.title(title)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((w, h)), 'gray')
        plt.subplots_adjust(hspace=0.001)
        plt.axis('off')


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
train_X = train_X.reshape((60000, 28 * 28))
train_Y = train_Y.reshape(60000)

test_X, test_Y = load_mnist('testing')
test_X = test_X.reshape((10000, 28 * 28))
test_Y = test_Y.reshape(10000)

print('train_X.shape', train_X.shape)
print('train_Y.shape', train_Y.shape)
# print('X_test.shape', test_X.shape)

w, h = 28, 28
n_features = 28 * 28
n_samples = len(train_X)

print("Total dataset size:")
print("train n_samples: %d" % n_samples)
print("total n_features: %d" % n_features)

# Data pre-processing: Normalization
train_X = sklearn.preprocessing.scale(train_X)
test_X = sklearn.preprocessing.scale(test_X)

# row_sums = train_X.sum(axis=1).astype(float)
# train_X = np.true_divide(train_X, row_sums[:, np.newaxis])
#
# row_sums = test_X.sum(axis=1).astype(float)
# test_X = np.true_divide(test_X, row_sums[:, np.newaxis])

###############################################################################
# Dictionary Learning
n_components = 700
n_samples_training = 20000

print("\nSparse Coding Dictionary Learning")
# pca = RandomizedPCA(n_components=n_dcomponents).fit(train_X)
# dl = KSVDSparseCoding(n_components, n_nonzero_coefs=70, preserve_dc=False, approx=False, max_iter=5, verbose=1)
# dl.fit(train_X[0:n_samples_training])

estimator = SparseFiltering(n_features=n_components,
                            maxfun=500,  # The maximal number of evaluations of the objective function
                            iprint=10)  # after how many function evaluations is information printed
# by L-BFGS. -1 for no information
features = estimator.fit_transform(train_X[0:n_samples_training])
print "features.shape", features.shape
print "estimator.w_.shape", estimator.w_.shape

plt.figure(figsize=(12, 10))
for i in range(estimator.w_.shape[0]):
    plt.subplot(int(np.sqrt(n_features)), int(np.sqrt(n_features)), i + 1)
    plt.pcolor(estimator.w_[i].reshape(w, h),
               cmap=plt.cm.gray, vmin=estimator.w_.min(),
               vmax=estimator.w_.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("")
plt.tight_layout()
plt.show()

print "dl.atom_bin_count", dl.atom_bin_count().shape, np.average(dl.atom_bin_count()), dl.atom_bin_count().tolist()

plt.plot(dl.errors)
plt.show()

print "X_train.shape", train_X.shape
print "Components shape", dl.dictionary.shape

# components = dl.components().reshape((n_components, n_features))
components = dl.dictionary

# Visualizing the components as images
plot_gallery("Visualizing top components", components.T, w, h, n_row=n_components / 10, n_col=10)
plt.show()

###############################################################################
# Sparse Encoding
print("\nSparse Encoding")
train_X_sc = dl.code.T
np.set_printoptions(precision=1, suppress=False, linewidth=800)

test_X_sc = np.zeros((len(test_X), n_components))
test_X_sc = dl.sparse_encode(test_X, components).T

print "train_X_sc.shape", train_X_sc.shape

###############################################################################
# Visualize reconstructed images
reconstructed_X = np.zeros((20, n_features))
reconstructed_X_idx = range(10)
reconstructed_X[reconstructed_X_idx] = test_X[reconstructed_X_idx]
reconstructed_X[reconstructed_X_idx] = np.dot(components, test_X_sc[:, reconstructed_X_idx])

print "reconstructed_X.shape", reconstructed_X.shape
print(reconstructed_X)

plot_gallery("Reconstructed images", reconstructed_X, w, h, n_row=2, n_col=10)
plt.show()

###############################################################################
# Train a neuralnet classification model

print("\nFitting the classifier to the training set")

# nnet = NeuralNetwork(lr=0.1, sizes=[25, 10, 5], seed=1234, n_epochs=50)
# nnet.initialize(n_components, 10, classes_mapping=np.arange(10))
clf = LinearSVC(verbose=True)
clf.fit(train_X_sc, train_Y[0:n_samples_training])

print("\nPredicting test examples with the trained classifier")
test_y_pred = clf.predict(test_X_sc)
print("\nPredicting training examples with the trained classifier")
train_y_pred = clf.predict(train_X_sc)

print test_y_pred
print "test_y_pred.shape", test_y_pred.shape

# i_iteration, i_loss, lr = clf.train(train_X_sc, np.arange(10))

###############################################################################
# Quantitative evaluation of the prediction accuracy on the test set

print("\nPredicting classes on the test set")
# train_y_pred, train_y_prob = nnet.predict(train_X_sc)
# test_y_pred, test_y_prob = nnet.predict(test_X_pca)
#
accuracy = 0.0
for i in range(len(train_Y)):
    if train_Y[i] == train_y_pred[i]:
        accuracy += 1
accuracy /= len(train_Y)

print("Classification accuracy = %f, among %d classes IN TRAINSET" % (accuracy, 10))

accuracy = 0.0
for i in range(len(test_Y)):
    print "test_Y[i]", test_Y[i], "test_y_pred[i]", test_y_pred[i]
    if test_Y[i] == test_y_pred[i]:
        accuracy += 1
accuracy /= len(test_Y)

print("Classification accuracy = %f, among %d classes IN TESTSET" % (accuracy, 10))

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
