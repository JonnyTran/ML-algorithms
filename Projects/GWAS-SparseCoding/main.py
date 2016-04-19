"""
author: Nhat Tran

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from neural_network import NeuralNetwork
from sklearn.decomposition import RandomizedPCA


class FisherLDA:
    def __init__(self, n_features, n_classes, classes_mapping, n_components=150):
        self.n_classes = n_classes
        self.n_features = n_features
        self.classes_mapping = classes_mapping
        self.n_components = n_components

    def fit(self, X, Y):
        self.n_samples = len(Y)
        self.total_mean = np.mean(X, axis=0)

        to_pca = np.linalg.inv(self.w_scatter(X, Y)) * self.b_scatter(X, Y)
        self.pca = RandomizedPCA(n_components=self.n_components).fit(to_pca)

    def b_scatter(self, X, Y):
        # Calculate means for each class
        class_means = np.zeros((self.n_classes, self.n_features))

        class_indices = {}
        for i in range(len(Y)):
            class_index = self.classes_mapping.index(Y[i])
            if class_indices.has_key(class_index):
                class_indices[class_index].append(i)
            else:
                class_indices[class_index] = [i, ]

        for i, c in enumerate(self.classes_mapping):
            class_means[i] = np.mean(X[class_indices[i]], axis=0)

        b_scatter = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_classes):
            b_scatter += np.outer(class_means[i] - self.total_mean, class_means[i] - self.total_mean)
        b_scatter /= self.n_classes

        return b_scatter

    def w_scatter(self, X, Y):
        # w_scatter = np.zeros((self.n_features, self.n_features))
        # for i in range(self.n_samples):
        #     w_scatter += np.outer(X[i] - self.total_mean, X[i] - self.total_mean)
        # w_scatter /= self.n_samples
        w_scatter = np.cov(X.T)

        return w_scatter

    def transform(self, x):
        return self.pca.transform(x)

    def components(self):
        return self.pca.components_


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
all_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
animal_labels = [2, 3, 4, 5]  #

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
print("total n_features: %d" % n_features)
print("total n_classes: %d" % n_classes)
print("n_classes to classify: %d" % len(animal_labels))

###############################################################################
# Compute a PCA on the dataset
# Use RandomizedPCA to more efficiently extract top n_components

n_components = 150

print("\nExtracting the Fisher LDA features")
# pca = RandomizedPCA(n_components=n_components).fit(train_X)
flda = FisherLDA(n_features=n_features, n_classes=len(animal_labels), classes_mapping=animal_labels)
flda.fit(train_X, train_y)

print "X_train.shape", train_X.shape
print "Components shape", flda.components().shape

components = flda.components().reshape((n_components, n_features))

print("\nProjecting the input data on the new feature space")
train_X_pca = np.zeros((len(train_X), n_components))
for i in range(len(train_X)):
    train_X_pca[i] = flda.transform(train_X[i])

test_X_pca = np.zeros((len(test_X), n_components))
for i in range(len(test_X)):
    test_X_pca[i] = flda.transform(test_X[i])

print "train_X_pca.shape", train_X_pca.shape

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
# Some visualization of learning progress, data point scatterplot, and principal
# components

def plot_gallery(title, images, titles, h, w, n_row=5, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.title(title)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((w, h, 3), order='F'), origin='lower')
        plt.title(titles[i], size=8)
        plt.xticks(())
        plt.yticks(())


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

# Visualizing the components as images
component_titles = ["component %d" % i for i in range(components.shape[0])]
plot_gallery("Visualizing top components", components, component_titles, w, h)
plt.show()
