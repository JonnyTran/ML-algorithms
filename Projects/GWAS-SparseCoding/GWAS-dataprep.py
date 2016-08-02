import h5py
import sklearn.preprocessing as preprocessing
from FeatureRepresentation.Multimodal import Multimodal
from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding
import matplotlib.pyplot as plt
import numpy as np

### Loading data matrices
gene_data = h5py.File('compact_data.mat', 'r')
cnv_data = h5py.File('cnv.mat', 'r')
dna_data = h5py.File('methy.mat', 'r')

gene = gene_data[u'G'][u'expression'][:].T
gene = preprocessing.scale(gene)

cnv = cnv_data[u'C'][u'cnv'][:].T
cnv = preprocessing.scale(cnv, with_std=False)

dna = dna_data[u'M'][u'methylation'][:].T
dna = preprocessing.scale(dna)


print "gene.shape", gene.shape
print "cnv.shape", cnv.shape
print "dna.shape", dna.shape

n_components = 70

# dl = DictionaryLearning(n_components, n_iter=400, n_jobs=4, verbose=2)
# dl = KSVDSparseCoding(n_components, max_iter=15, verbose=1)
# dl.fit(cnv)

# mm = Multimodal(n_modals=2, sparse_coder=KSVDSparseCoding(n_components=n_components, max_iter=20, verbose=1))
# mm.fit([gene, cnv])

# plt.plot(mm.errors)
# plt.show()

# components = mm.dictionary
# codes = mm.code.T

# for i in range(components.shape[1]):
# print i, components[i].shape, np.mean(components[i]), np.linalg.norm(components[i])

# print "\n\ncomponents", components
# print "\n\ncodes", codes
