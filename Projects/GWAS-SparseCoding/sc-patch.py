"""
author: Nhat Tran

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import sparse_encode, MiniBatchDictionaryLearning
from sklearn.feature_extraction import image

from NeuralNetworks.neural_network import NeuralNetwork


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def plot_gallery(title, images, titles, h, w, n_row=10, n_col=10):
    plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))
    plt.title(title)
    # plt.subplots_adjust(bottom=0.1, left=.01, right=.99, top=.90, hspace=.1)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((w, h, 3), order='F'), origin='lower')
        plt.title(titles[i], size=8)
        plt.xticks(())
        plt.yticks(())


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
all_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
animal_labels = [0]  #

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
print("test n_samples: %d" % len(test_X))
print("total n_features: %d" % n_features)
print("total n_classes: %d" % n_classes)
print("n_classes to classify: %d" % len(animal_labels))

###############################################################################
# Patch Extraction
n_patches_ea_pic = 10
patch_w, patch_h = 8, 8
n_features = patch_w * patch_h * 3

train_X_patches = np.zeros(n_patches_ea_pic * n_samples * n_features) \
    .reshape(n_patches_ea_pic * n_samples, n_features)

for i in range(n_samples):
    train_X_img = train_X[i].reshape((32, 32, 3))
    train_X_img_patches = image.extract_patches_2d(train_X_img, (patch_w, patch_h), max_patches=n_patches_ea_pic,
                                                   random_state=0)

    train_X_patches[i * n_patches_ea_pic: (i + 1) * n_patches_ea_pic] = train_X_img_patches.reshape(n_patches_ea_pic,
                                                                                                    patch_w * patch_h * 3)

print "train_X_patches", train_X_patches.shape

###############################################################################
# Dictionary Learning
n_components = 576

print("\nSparse Coding Dictionary Learning")
# pca = RandomizedPCA(n_components=n_components).fit(train_X)
dl = MiniBatchDictionaryLearning(n_components)
dl.fit(train_X_patches)

print "X_train.shape", train_X.shape
print "Components shape", dl.components_.shape

# components = dl.components().reshape((n_components, n_features))
components = dl.components_

# Visualizing the components as images
component_titles = ["component %d" % i for i in range(components.shape[0])]
plot_gallery("Visualizing top components", components, component_titles, patch_w, patch_h, n_row=24, n_col=24)
plt.show()

###############################################################################
# Sparse Encoding
print("\nSparse Encoding")
train_X_pca = np.zeros((len(train_X), n_components))
train_X_pca = sparse_encode(train_X, components, algorithm='omp')
np.set_printoptions(precision=3, suppress=True)
print train_X_pca
# for i in range(len(train_X)):
#     train_X_pca[i] = dl.transform(train_X[i])

test_X_pca = np.zeros((len(test_X), n_components))
test_X_pca = sparse_encode(test_X, components, algorithm='omp')
# for i in range(len(test_X)):
#     test_X_pca[i] = dl.transform(test_X[i])

print "train_X_pca.shape", train_X_pca.shape

###############################################################################
# Visualize reconstructed images
reconstructed_X = np.zeros((20, n_features))
reconstructed_X[0:10] = train_X[0:10]
reconstructed_X[10:20] = np.dot(train_X_pca[0:10], components)

reconstructed_titles = ["reconstructed %d" % i for i in range(len(reconstructed_X))]
plot_gallery("Reconstructed images", reconstructed_X, reconstructed_titles, w, h, 2, 10)
plt.show()

###############################################################################
# Train a neuralnet classification model

print("\nFitting the neural net to the training set")

nnet = NeuralNetwork(lr=0.1, sizes=[50, 25, 10], seed=1234, n_epochs=50)
nnet.initialize(n_components, len(animal_labels), classes_mapping=animal_labels)

i_iteration, i_loss, lr = nnet.train(train_X_pca, train_y)

###############################################################################
# Quantitative evaluation of the prediction accuracy on the test set

print("\nPredicting classes on the test set")
train_y_pred, train_y_prob = nnet.predict(train_X_pca)
test_y_pred, test_y_prob = nnet.predict(test_X_pca)

accuracy = 0.0
for i in range(len(train_y)):
    if train_y[i] == train_y_pred[i]:
        accuracy += 1
accuracy /= len(train_y)

print("Classification accuracy = %f, among %d classes IN TRAINSET" % (accuracy, len(animal_labels)))

accuracy = 0.0
for i in range(len(test_y)):
    if test_y[i] == test_y_pred[i]:
        accuracy += 1
accuracy /= len(test_y)

print("Classification accuracy = %f, among %d classes IN TESTSET" % (accuracy, len(animal_labels)))

###############################################################################
# Some visualization of learning progress, data point scatterplot

# Visualize neural net learning
plt.figure()
plt.title("NNet Learning (Error vs. Epoch). Rate: %0.5f, Accuracy: %0.3f" % (lr, accuracy))
plt.xlabel("# of epoch iteration")
plt.ylabel("Training loss (negative log)")
plt.plot(i_iteration, i_loss, marker='.')

# Visualize data points on top principle components on scatterplot
scatterplot = plt.figure()
ax = Axes3D(scatterplot)  # use the plotting figure to create a Axis3D object.
colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
for i, animal in enumerate(animal_labels):
    animal_indices = []
    for j in range(len(train_y)):
        if train_y[j] == animal: animal_indices.append(j)

    ax.scatter(train_X_pca[animal_indices][0], train_X_pca[animal_indices][1], train_X_pca[animal_indices][2],
               color=colors[i % len(colors)], label=all_labels[animal])

ax.set_xlabel("1st principal component")
ax.set_ylabel("2nd principal component")
ax.set_zlabel("3rd principal component")
ax.set_title("Scatter plot of all images and their classes")
