from time import time

from sklearn.decomposition import RandomizedPCA


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


###############################################################################
# Load the data

label_names = unpickle('./cifar-10-batches-py/batches.meta')
train_files = ['./cifar-10-batches-py/data_batch_1', './cifar-10-batches-py/data_batch_2',
               './cifar-10-batches-py/data_batch_3', './cifar-10-batches-py/data_batch_4',
               './cifar-10-batches-py/data_batch_5']

test_batch = unpickle('./cifar-10-batches-py/test_batch')
data_batch_1 = unpickle('./cifar-10-batches-py/data_batch_1')

X_train = data_batch_1['data']
X_test = data_batch_1['labels']
Y_train = test_batch['data']
Y_test = test_batch['labels']

print("Total dataset size:")
print("train n_samples: %d" % len(X_train))
print("test n_samples: %d" % len(Y_train))
print("n_features: %d" % len(X_train[0]))
print("n_classes: %d" % len(label_names))

###############################################################################
# Compute a PCA (eigenfaces) on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenvectors from %d images"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print pca
# eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
