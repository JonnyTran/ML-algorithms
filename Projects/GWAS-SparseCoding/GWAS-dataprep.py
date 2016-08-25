import h5py
import sklearn.preprocessing as preprocessing
from sklearn.cross_validation import train_test_split
from FeatureRepresentation.Multimodal import Multimodal
from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding
import matplotlib.pyplot as plt
import numpy as np

### Loading data matrices
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

print "gene.shape", gene.shape
print "snp.shape", snp.shape
print "cnv.shape", cnv.shape
print "dna.shape", dna.shape
print "phenotype.shape", phenotype.shape

### Integrating data
data = [gene, snp, cnv, dna]
n_samples = data[0].shape[0]
n_features = [0, ] * len(data)
for i in range(len(data)):
    n_features[i] = data[i].shape[1]
    # X_s[i] /= n_features[i] ** 0.5  # Scale constant

print "X_s: n_samples", n_samples, ", n_features", n_features
X_s = np.concatenate(data, axis=1)

print "X_s.shape", X_s.shape

### Splitting into training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X_s, phenotype, test_size=0.25, random_state=42)

print "X_train.shape", X_train.shape
print "X_test.shape", X_test.shape

### Sparse Representation
n_components = 5

# dl = DictionaryLearning(n_components, n_iter=400, n_jobs=4, verbose=2)
dl = KSVDSparseCoding(n_components, max_iter=15, verbose=1)
dl.fit(X_s)
#
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
