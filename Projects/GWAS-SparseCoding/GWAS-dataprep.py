import h5py
from sklearn.decomposition import sparse_encode, DictionaryLearning

# a = sio.loadmat('compact_data.mat')

gene_data = h5py.File('compact_data.mat', 'r')
cnv_data = h5py.File('cnv.mat', 'r')
dna_data = h5py.File('methy.mat', 'r')

gene = gene_data[u'G'][u'expression'][:].T
# cnv = cnv_data[u'C'][u'cnv'][:].T
# dna = dna_data[u'M'][u'methylation'][:].T

n_components = 300

dl = DictionaryLearning(n_components, n_iter=400, n_jobs=4, verbose=2)
dl.fit(gene)
components = dl.components_

train_X_sc = sparse_encode(gene, components)
