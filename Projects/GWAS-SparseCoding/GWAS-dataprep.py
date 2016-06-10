import h5py
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding

# a = sio.loadmat('compact_data.mat')

gene_data = h5py.File('compact_data.mat', 'r')
# cnv_data = h5py.File('cnv.mat', 'r')
# dna_data = h5py.File('methy.mat', 'r')

gene = gene_data[u'G'][u'expression'][:].T
gene = preprocessing.scale(gene)
# cnv = cnv_data[u'C'][u'cnv'][:].T
# dna = dna_data[u'M'][u'methylation'][:].T

print "gene.shape", gene.shape

n_components = 60

# dl = DictionaryLearning(n_components, n_iter=400, n_jobs=4, verbose=2)
dl = KSVDSparseCoding(n_components, max_iter=25, verbose=1)
dl.fit(gene)

plt.plot(dl.errors)
plt.show()

components = dl.dictionary

codes = dl.code.T

print "\n\ncomponents", components
print "\n\ncodes", codes
