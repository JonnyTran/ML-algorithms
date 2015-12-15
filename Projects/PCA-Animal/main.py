from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


###############################################################################
# Load the data

train_files = ['./cifar-10-batches-py/data_batch_1', './cifar-10-batches-py/data_batch_2',
               './cifar-10-batches-py/data_batch_3', './cifar-10-batches-py/data_batch_4',
               './cifar-10-batches-py/data_batch_5']

test_batch = unpickle('./cifar-10-batches-py/test_batch')
data_batch_1 = unpickle('./cifar-10-batches-py/data_batch_1')
label_names = unpickle('./cifar-10-batches-py/batches.meta')
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
animal_labels = [2, 3, 4, 5, 6, 7]
print label_names

X_train = data_batch_1['data']
Y_train = data_batch_1['labels']
X_test = test_batch['data']
Y_test = test_batch['labels']

w, h = 32, 32
r, g, b = 1024, 1024, 1024
n_features = 3072
n_samples = len(X_train)
n_classes = len(label_names['label_names'])
target_names = label_names['label_names']

print("Total dataset size:")
print("train n_samples: %d" % n_samples)
print("test n_samples: %d" % len(Y_train))
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Compute a PCA on the dataset
# Use RandomizedPCA to more efficiently extract top n_components

n_components = 150

print("\nExtracting the top %d eigenvectors from %d images"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print "X_train.shape", X_train.shape
print "pca.components_.shape", pca.components_.shape

components = pca.components_.reshape((n_components, n_features))
print "pca.components_.reshape", pca.components_.shape

print("\nProjecting the input data on the eigenvectors orthonormal basis")
t0 = time()
X_train_pca = np.zeros((n_samples, n_components))
for i in range(len(X_train)):
    X_train_pca[i] = pca.transform(X_train[i])

X_test_pca = np.zeros((n_samples, n_components))
for i in range(len(X_test)):
    X_train_pca[i] = pca.transform(X_test[i])

print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

# print("\nFitting the classifier to the training set")
# t0 = time()
# clf = SVC(kernel='rbf', class_weight='balanced')
# clf = clf.fit(X_train_pca, Y_train)
# print("done in %0.3fs" % (time() - t0))
# print(clf)


###############################################################################
# Quantitative evaluation of the model quality on the test set

# print("\nPredicting classes on the test set")
# t0 = time()
# y_pred = clf.predict(X_test_pca)
# print("done in %0.3fs" % (time() - t0))
#
# print(classification_report(Y_test, y_pred))
# print(confusion_matrix(Y_test, y_pred, labels=range(n_classes)))



###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=10, n_col=15):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.10)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w, 3), order='F'), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


# prediction_titles = [title(y_pred, Y_test, target_names, i)
#                      for i in range(y_pred.shape[0])]
#
# plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative components
component_titles = ["component %d" % i for i in range(components.shape[0])]
plot_gallery(components, component_titles, h, w)

plt.show()
