"""
author: Nhat Tran

Copied code structure from http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from NeuralNetworks.neural_network import NeuralNetwork


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


###############################################################################
# Load the data

test_batch = unpickle('./cifar-10-batches-py/test_batch')
data_batch_1 = unpickle('./cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('./cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('./cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('./cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('./cifar-10-batches-py/data_batch_5')
label_names = unpickle('./cifar-10-batches-py/batches.meta')

# Concatenate all 5 batches of data
train_X = data_batch_1['data']
train_X = np.vstack((train_X, data_batch_2['data']))
train_X = np.vstack((train_X, data_batch_3['data']))
train_X = np.vstack((train_X, data_batch_4['data']))
train_X = np.vstack((train_X, data_batch_5['data']))

train_y = data_batch_1['labels']
train_y = np.hstack((train_y, data_batch_2['labels']))
train_y = np.hstack((train_y, data_batch_3['labels']))
train_y = np.hstack((train_y, data_batch_4['labels']))
train_y = np.hstack((train_y, data_batch_5['labels']))

test_X = test_batch['data']
test_y = np.array(test_batch['labels'])

# Subset the data to only animal labels
animal_labels = [3, 7]  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_subset_indices = []
for i in range(len(train_y)):
    if train_y[i] in animal_labels: train_subset_indices.append(i)
train_X = train_X[train_subset_indices]
train_y = train_y[train_subset_indices]

test_subset_indices = []
for i in range(len(test_y)):
    if test_y[i] in animal_labels: test_subset_indices.append(i)
test_X = test_X[test_subset_indices]
test_y = test_y[test_subset_indices]

print('X_train.shape', train_X.shape)
print('X_test.shape', test_X.shape)

w, h = 32, 32
n_features = 3072
n_samples = len(train_X)
n_classes = len(label_names['label_names'])
target_names = label_names['label_names']

print("Total dataset size:")
print("train n_samples: %d" % n_samples)
print("test n_samples: %d" % len(train_y))
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Compute a PCA on the dataset
# Use RandomizedPCA to more efficiently extract top n_components

n_components = 150

print("\nExtracting the top %d eigenvectors from %d images"
      % (n_components, train_X.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components).fit(train_X)
print("done in %0.3fs" % (time() - t0))

print "X_train.shape", train_X.shape
print "pca.components_.shape", pca.components_.shape

components = pca.components_.reshape((n_components, n_features))
print "pca.components_.reshape", pca.components_.shape

print("\nProjecting the input data on the eigenvectors orthonormal basis")
t0 = time()
train_X_pca = np.zeros((n_samples, n_components))
for i in range(len(train_X)):
    train_X_pca[i] = pca.transform(train_X[i])

test_X_pca = np.zeros((n_samples, n_components))
for i in range(len(test_X)):
    train_X_pca[i] = pca.transform(test_X[i])

print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a neuralnet classification model

print("\nFitting the neural net to the training set")
t0 = time()

nnet = NeuralNetwork(lr=1e-8,
                     dc=1e-10,
                     sizes=[100, 50, 25],
                     L2=0.001,
                     L1=0,
                     seed=1234,
                     tanh=False,
                     n_epochs=100)
nnet.initialize(n_components, len(animal_labels), classes_mapping=animal_labels)
nnet.train(train_X_pca, train_y)

print("done in %0.3fs" % (time() - t0))
print(nnet)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("\nPredicting classes on the test set")
t0 = time()
y_pred = nnet.predict(test_X_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(test_y, y_pred))
print(confusion_matrix(test_y, y_pred, labels=range(n_classes)))



###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=10, n_col=15):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((w, h, 3), order='F'), origin='lower')
        plt.title(titles[i], size=8)
        plt.xticks(())
        plt.yticks(())


plot_gallery(test_X, [x for x in range(25)], h, w, n_row=3, n_col=4)

# the most significative components as images
component_titles = ["component %d" % i for i in range(components.shape[0])]
plot_gallery(components, component_titles, h, w)

plt.show()
