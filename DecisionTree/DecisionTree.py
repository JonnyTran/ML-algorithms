from copy import deepcopy

import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, stopping_criteria=1.0):
        self.stopping_criteria = stopping_criteria
        self.decision_tree = None;

    def train(self, trainset, target_attr):
        """

        :param dataset: a panda dataframe object
        :param target_attr: a string of the name of the field containing class
        """

        ##### Initialize dataset variables #####
        self.trainset = trainset
        self.trainset_size = len(trainset)
        self.target_attr = target_attr
        self.classes = trainset[target_attr].unique()
        self.n_classes = len(self.classes)
        self.split_attrs = list(self.trainset.columns.difference([target_attr]))

        ##### Train Decision Tree #####
        self.decision_tree = self.create_decision_tree(self.split_attrs, target_attr, [], self.classes[0])
        # print self.decision_tree

    def create_decision_tree(self, split_attrs, target_attr, conditions, best_guess_so_far):
        """
        This builds the tree by recursively finding the next best attribute to split (lowest entropy)

        :param split_attrs: Attributes available left to split, in this subtree
        :param target_attr: The class attribute
        :param conditions: A list of conditions splits on attributes in a tree (The depth of the tree)
        :param best_guess_so_far: The most popular class label before splitting
        :return:  Return a recursive subtree
        """
        entropy, best_guess_so_far, partition_size, conditions_list = \
            self.compute_entropy(conditions, target_attr, best_guess_so_far)
        # print "split:",entropy, best_guess_so_far, partition_size, conditions_list
        if (entropy == 0.0 or
                    best_guess_so_far[1] >= self.stopping_criteria or
                    len(split_attrs) == 0 or
                    len(conditions) > 29):
            return best_guess_so_far[0]

        best_next_split_attr = self.find_best_split_attr(split_attrs, target_attr, conditions, best_guess_so_far)
        # print "best split:", best_next_split_attr
        tree = {best_next_split_attr: {}}

        for value in self.trainset[best_next_split_attr].unique():
            conditions_to_split = deepcopy(conditions)
            conditions_to_split.append((best_next_split_attr, value))
            subtree = self.create_decision_tree(set(split_attrs) - set(best_next_split_attr),
                                                target_attr,
                                                conditions_to_split,
                                                best_guess_so_far)
            tree[best_next_split_attr][value] = subtree

        return tree

    def find_best_split_attr(self, split_attrs, target_attr, conditions, best_guess_so_far):
        """
        Try splitting on all split_attrs, and pick the one with lowest entropy

        :param split_attrs: Attributes available left to split, in this subtree
        :param target_attr: The class attribute
        :param conditions: A list of conditions splits on attributes in a tree (The depth of the tree)
        :param best_guess_so_far: The most popular class label before splitting
        :return:  Return a recursive subtree
        """
        best_gain = np.inf
        best_split_attr = None

        for attribute in split_attrs:
            entropy_gain_sum = 0.0
            info_split_sum = 0.0
            split_gain = 0.0
            field_values = list(np.unique(self.trainset[attribute]))

            for value in field_values:
                condition_to_split = deepcopy(conditions)
                condition_to_split.append((attribute, value))

                entropy, most_popular_class, partition_size, conditions_list = \
                    self.compute_entropy(condition_to_split, target_attr, best_guess_so_far)

                entropy_gain_sum += entropy
            # info_split_sum += -np.true_divide(partition_size, self.trainset_size) *\
            #                       np.log2(np.true_divide(partition_size, self.trainset_size))
            #
            # split_gain = np.true_divide(entropy_gain_sum, info_split_sum)

            # print "attribute:", attribute, ", split_gain:", split_gain
            # if (split_gain >= best_gain):
            #     best_gain = split_gain
            #     best_split_attr = attribute

            if (best_gain >= entropy_gain_sum):
                best_gain = entropy_gain_sum
                best_split_attr = attribute

        return best_split_attr

    def compute_entropy(self, conditions, target_attr, best_guess_so_far, gain_ratio=False):
        """
        Computes the entropy of a partitioned dataset, and do popular vote

        :param trainset: The dataset
        :param conditions: An array of conditions to filter the trainset
        :param target_attr: Name of the column containing class label
        :param gain_ratio: Use SplitInfo to penalize large number of partitions (if True) (default False)
        :return: the entropy of splitting the trainset, the most popular class label, and count of the partition
        """
        filtered_set = self.trainset

        filtered_set = self.filter_dataset(filtered_set, conditions)

        if (len(filtered_set) == 0): return 0.0, best_guess_so_far, 1, conditions

        sum_entropy = 0.0
        most_popular_class = ''
        class_counts = self.count_classes(filtered_set, target_attr)
        for i, c in enumerate(class_counts):
            if (i == 0): most_popular_class = c
            sum_entropy += -c[1] * np.log2(c[1])

        return sum_entropy, \
               most_popular_class, \
               len(filtered_set), \
               conditions

    def filter_dataset(self, filtered_set, conditions):
        """
        Query the dataset by applying conditions on attributes (e.g. a=="1", b=="2", etc.)

        :param filtered_set:
        :param conditions: A list of conditions
        :return:
        """
        conditions_list = []
        if len(conditions) > 0:
            query_str = ''
            for i, condition in enumerate(conditions):
                query_str += condition[0] + ' == "' + condition[1] + '"'
                conditions_list.append(condition[0] + ' == "' + condition[1] + '"')
                if (i < len(conditions) - 1): query_str += ' and '

            filtered_set = filtered_set.query(query_str)

        return filtered_set

    def count_classes(self, filtered_set, target_attr, normalize=True):
        class_counts = filtered_set[target_attr].value_counts(sort=True, ascending=False,
                                                              normalize=normalize).iteritems()
        return class_counts

    def predict(self, test_tuple):
        if (type(self.decision_tree) is str):
            test_tuple["prediction"] = self.decision_tree
        else:
            self.predict_recursive_helper(test_tuple, self.decision_tree)
        return test_tuple

    def predict_recursive_helper(self, test_tuple, subtree):
        for attr, subtree in subtree.items():
            for attr_value, subtree_2 in subtree.items():
                if test_tuple[attr] == attr_value:
                    if type(
                            subtree_2) is str:  # If subtree_2 is a leaf node of the decision tree, and contain the class prediction
                        test_tuple["prediction"] = subtree_2
                        return
                    else:
                        self.predict_recursive_helper(test_tuple, subtree_2)

    def test(self, testset, target_attr):
        testset_size = len(testset)
        testset["prediction"] = None
        testset.apply(self.predict, axis=1)
        print testset
        return np.true_divide(len(testset[testset[target_attr] == testset["prediction"]]), testset_size)


def main():
    trainset = pd.read_csv("MushroomTrain.csv")
    trainset.drop('a', axis=1, inplace=True)

    testset = pd.read_csv("MushroomTest.csv")
    testset.drop('p.1', axis=1, inplace=True)
    testset.columns = ['e', 'x', 's', 'y', 't']

    decision_tree = DecisionTreeClassifier()
    decision_tree.train(trainset, target_attr='e')
    print "Accuracy on testset:", decision_tree.test(testset, target_attr='e')
    print decision_tree.decision_tree


    # TODO Compare accuracy on test data vs training data

    # full_mushroom_dataset = pd.read_csv("agaricus-lepiota.data")
    # full_mushroom_dataset.drop('p2', axis=1, inplace=True)
    # full_mushroom_dataset.drop('p1', axis=1, inplace=True)
    #
    # trainset = full_mushroom_dataset.sample(n=6123, replace=False, random_state=1234)
    # testset = full_mushroom_dataset.sample(n=2000, replace=False, random_state=5678)
    #
    # decision_tree = DecisionTreeClassifier()
    # decision_tree.train(trainset, target_attr='p')
    # print decision_tree.test(testset, target_attr='p')


if __name__ == "__main__":
    main()
