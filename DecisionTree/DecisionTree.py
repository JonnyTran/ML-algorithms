from collections import defaultdict

import numpy as np


class DecisionTree:
    def __init__(self):
        pass


class DecisionTreeClassifier:
    def __init__(self, max_level=2):
        self.max_level = max_level

        def hash(): return defaultdict(hash)

        self.decision_tree = hash()

    def train(self, trainset, class_field):
        """

        :param dataset: a panda dataframe object
        :param class_field: a string of the name of the field containing class
        """

        ##### Initialize dataset variables #####
        self.trainset = trainset
        self.input_size = len(trainset)
        self.classes = list(np.unique(trainset[class_field]))
        self.n_classes = len(self.classes)
        self.split_fields = list(self.trainset.columns - [class_field])

        ##### Initialize Decision Tree #####


        for level in range(self.max_level):
            # Find best split
            for field in self.split_fields:
                values = list(np.unique(trainset[field]))

    def computeEntropy(self, trainset, conditions, class_field, n_classes):
        pass

    def split(self):
        pass

    def entropy(self, node_data):
        pass

    def test(self, testset):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
