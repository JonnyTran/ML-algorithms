import h5py
import sklearn.preprocessing as preprocessing
from sklearn.cross_validation import train_test_split
from FeatureRepresentation.Multimodal import Multimodal
from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning, sparse_encode, MiniBatchDictionaryLearning
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import numpy as np

###############################################################################
# Loading data matrices
gene_data = h5py.File('compact_data.mat', 'r')
cnv_data = h5py.File('cnv.mat', 'r')
dna_data = h5py.File('methy.mat', 'r')

gene = gene_data[u'G'][u'expression'][:].T
gene = np.delete(gene, (56), axis=0)
gene = preprocessing.scale(gene)

snp = gene_data[u'S'][u'additive_encoding'][:].T
snp = np.delete(snp, (56), axis=0)
snp = preprocessing.scale(snp)

cnv = cnv_data[u'C'][u'cnv'][:].T
cnv = np.delete(cnv, (56), axis=0)
cnv = preprocessing.scale(cnv, with_std=False)

dna = dna_data[u'M'][u'methylation'][:].T
dna = preprocessing.scale(dna)

phenotype = gene_data[u'S'][u'phenotype'][:].T
phenotype = np.delete(phenotype, (56), axis=0)
phenotype = phenotype.astype(int).reshape(-1)

n_classes = 4

print "gene.shape", gene.shape
print "snp.shape", snp.shape
print "cnv.shape", cnv.shape
print "dna.shape", dna.shape

###############################################################################
# Integrating data
data = [cnv, dna]
n_samples = data[0].shape[0]
n_features = [0, ] * len(data)
for i in range(len(data)):
    n_features[i] = data[i].shape[1]
    # X_s[i] /= n_features[i] ** 0.5  # Scale constant

print "X_s: n_samples", n_samples, ", n_features", n_features
X_s = np.concatenate(data, axis=1)
X_s = preprocessing.scale(X_s)

print "X_s.shape", X_s.shape

###############################################################################
# Splitting into training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X_s, phenotype, test_size=0.25, random_state=42)

print "X_train.shape", X_train.shape
print "y_train", y_train
print "X_test.shape", X_test.shape
print "y_test", y_test

###############################################################################
# Write data to matlab matrix
# scipy.io.savemat('/home/jonny2/PycharmProjects/ML-algorithms/Projects/GWAS-SparseCoding/psychiatric.mat',
#                  mdict={'tr_dat': X_train, 'tt_dat': X_test, 'trls': y_train, 'ttls': y_test})


###############################################################################
# Sparse Representation
n_components = 25

# dl = DictionaryLearning(n_components, max_iter=15, n_jobs=4, verbose=2)
dl = KSVDSparseCoding(n_components, max_iter=5, verbose=1, approx=True)
dl.fit(X_s)

eigenfaces = dl.components_.T

print("Projecting the input data on the learned dictionary bases")
X_train_pca = sparse_encode(X_train, eigenfaces, algorithm='lasso_lars')
X_test_pca = sparse_encode(X_test, eigenfaces, algorithm='lasso_lars')

print "X_train_pca.shape", X_train_pca.shape
print "X_test_pca.shape", X_test_pca.shape

###############################################################################
# Train a SVM classification model
print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

###############################################################################
# Quantitative evaluation of the model quality on the test set
print("Predicting disease phenotype on the test set")
y_pred = clf.predict(X_test_pca)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# mm = Multimodal(n_modals=2, sparse_coder=KSVDSparseCoding(n_components=n_components, max_iter=20, verbose=1))
# mm.fit([dna, cnv])
#
# plt.plot(mm.errors)
# plt.show()
#
# components = mm.dictionary
# codes = mm.code.T
#
# for i in range(components.shape[1]):
#     print i, components[i].shape, np.mean(components[i]), np.linalg.norm(components[i])
#
# print "\n\ncomponents", components
# print "\n\ncodes", codes
