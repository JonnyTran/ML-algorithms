import numpy as np
import pandas as pd

from ML_Algorithms.DecisionTree import DecisionTree


class EnsembleBagging():
    def __init__(self, classifier, n_bagging=10):
        self.ensemble_classifiers = []

        for i in range(n_bagging):
            aClassifier = classifier()
            self.ensemble_classifiers.append(aClassifier)

    def train(self, trainset, target_attr, partition_ratio=1.0, replace=True, seed=1234):
        """

        :param trainset:
        :param target_attr:
        :param partition_ratio:
        :param replace:
        """
        partition_size = np.ceil(partition_ratio * len(trainset))

        for i, classifier in enumerate(self.ensemble_classifiers):
            sampling_seed = seed + i
            trainset_partition = trainset.sample(n=partition_size, replace=replace, random_state=sampling_seed)

            classifier.train(trainset_partition, target_attr)

    def predict(self, test_tuple):
        predictions = []
        for classifier in self.ensemble_classifiers:
            prediction = classifier.predict(test_tuple)["prediction"]
            if prediction != None:
                predictions.append(prediction)

        u, indices = np.unique(predictions, return_inverse=True)
        popular_vote = u[np.argmax(np.bincount(indices))]

        test_tuple["prediction"] = popular_vote
        return test_tuple

    def test(self, testset, target_attr):
        testset_size = len(testset)
        testset["prediction"] = None
        testset.apply(self.predict, axis=1)

        return np.true_divide(len(testset[testset[target_attr] == testset["prediction"]]), testset_size)


def main():
    trainset = pd.read_csv("../DecisionTree/MushroomTrain.csv")
    trainset.drop('a', axis=1, inplace=True)

    testset = pd.read_csv("../DecisionTree/MushroomTest.csv")
    testset.drop('p.1', axis=1, inplace=True)
    testset.columns = ['e', 'x', 's', 'y', 't']

    for i in [10, 50, 100]:
        print "Training", i, "decision trees"
        tree_bagging_ensemble = EnsembleBagging(DecisionTree.DecisionTreeClassifier, n_bagging=i)
        tree_bagging_ensemble.train(trainset, target_attr='e', partition_ratio=1.0, seed=i)
        print "Accuracy on testset", tree_bagging_ensemble.test(testset, target_attr='e')



if __name__ == '__main__':
    main()
