from collections import defaultdict

import pandas as pd
import numpy as np


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

        self.computeEntropy(self.trainset, [('e', 'p'), ('x', 'x'), ('s', 'y'), ('y', 'n')], self.classes)

        # for level in range(self.max_level):
        #     # Find best split
        #     for field in self.split_fields:
        #         values = list(np.unique(trainset[field]))

    def computeEntropy(self, trainset, conditions, classes):
        filtered_set = trainset

        if len(conditions) > 0:
            query_str = ''
            for i, condition in enumerate(conditions):
                query_str += condition[0] + ' == "' + condition[1] + '"'
                if (i < len(conditions) - 1): query_str += ' and '

            filtered_set = filtered_set.query(query_str)

        counts = filtered_set['e'].value_counts(sort=True, ascending=False, normalize=True)


    def split(self):
        pass

    def entropy(self, node_data):
        pass

    def test(self, testset):
        pass


def main():
    tree = DecisionTreeClassifier()
    trainset = pd.read_csv("MushroomTrain.csv")
    trainset.drop('a', axis=1, inplace=True)
    tree.train(trainset, class_field='e')


if __name__ == "__main__":
    main()
