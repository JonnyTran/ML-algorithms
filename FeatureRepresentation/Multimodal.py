import numpy as np

from FeatureRepresentation.SparseRepresentation import KSVDSparseCoding


class Multimodal():
    def __init__(self, n_modals, sparse_coder):
        self.n_modals = n_modals
        self.sparse_coder = sparse_coder

    def fit(self, X_s):
        self.initialize(X_s)
        self.sparse_coder.fit(self.X_s)

        self.dictionaries = []
        features_indx = np.cumsum(self.n_features)
        for i in range(self.n_modals):
            if i == 0:
                self.dictionaries.append(
                    self.sparse_coder.dictionary[0:features_indx[i], :] * self.n_features[i] ** 0.5)
            else:
                a = features_indx[i - 1]
                b = features_indx[i]
                self.dictionaries.append(
                    self.sparse_coder.dictionary[a:b, :] * self.n_features[i] ** 0.5)

    def initialize(self, X_s):
        self.n_samples = X_s[0].shape[0]
        self.n_features = [0, ] * len(X_s)
        for i in range(len(X_s)):
            self.n_features[i] = X_s[i].shape[1]
            # X_s[i] /= self.n_features[i] ** 0.5  # Scale constant

        print "X_s: n_samples", self.n_samples, ", n_features", self.n_features
        self.X_s = np.concatenate(X_s, axis=1)

    def get_dictionary(self, modal_no):
        return self.dictionaries[modal_no]

    def sparse_encode(self, X, modal_no):
        return self.sparse_coder.sparse_encode(X, self.dictionaries[modal_no])


def main():
    n_samples = 500
    n_components = 50
    n_features = [20000, 10000]

    X0 = np.random.rand(n_samples, n_features[0]) - 0.5
    X1 = np.random.rand(n_samples, n_features[1]) - 0.5

    print "X0", X0
    print "X1", X1

    # Last 10 of X0 and first 10 of X1 are shared by: X0 + 0.3X1 + 0.1*U(-0.5, 0.5)
    X0[:, 0:10] += 0.3 * X1[:, 0:10] + 0.1 * (np.random.rand(n_samples, 10) - 0.5)
    X1[:, 0:10] += X0[:, 0:10] + 0.1 * (np.random.rand(n_samples, 10) - .05)

    mm = Multimodal(n_modals=2, sparse_coder=KSVDSparseCoding(n_components=n_components, max_iter=20, verbose=1))
    mm.fit([X0, X1])

    for dic in mm.dictionaries:
        print dic.shape
        print "dic[:,0] norm", np.linalg.norm(dic[:, 1])
        print dic[:, 1]


if __name__ == "__main__":
    main()
