import heapq
import itertools

import numpy as np
import pandas as pd


class HierarchicalCluster:
    SINGLE_LINKAGE = 0
    COMPLETE_LINKAGE = 1
    AVERAGE_LINKAGE = 2  # TODO

    def __init__(self, linkage=COMPLETE_LINKAGE):
        self.linkage = linkage
        self.dist_adj_list = []
        self.hier_clusters = []  # Stores the clusters at each iteration of the clustering algorithm

    def train(self, dataset):
        self.dataset = dataset
        self.dataset_size = len(dataset)

        self.build_initial_dist_matrix(range(self.dataset_size))
        self.hier_clusters.append(range(self.dataset_size))

        i = 0
        while self.dist_adj_list:
            # print i, self.dist_adj_list
            i += 1
            self.merge_next()

            # for c in self.hier_clusters:
            #     print c

    def build_initial_dist_matrix(self, data_points):
        pairs = list(itertools.combinations(data_points, 2))
        for pair in pairs:
            self.dist_adj_list.append(
                (self.euclidean_dist(self.dataset.ix[pair[0]], self.dataset.ix[pair[1]]),
                 {pair[0], pair[1]})
            )
        heapq.heapify(self.dist_adj_list)

    def merge_next(self):
        print self.dist_adj_list[0]
        new_cluster = self.dist_adj_list.pop(0)[1]

        self.update_hier_clusters(new_cluster)
        self.update_dist_adj_list(new_cluster)

    def update_hier_clusters(self, new_cluster):
        groups = [x for x in self.hier_clusters[-1]
                  if (not (type(x) is int and x in new_cluster))
                  and (not ((type(x) is set or type(x) is frozenset) and new_cluster.issuperset(x)))]

        groups.append(new_cluster)
        self.hier_clusters.append(groups)

    def update_dist_adj_list(self, new_cluster):
        """
        With a new cluster formed, this function will update the distance adjacency list that contain distance of
        every other cluster to this new cluster.
        The distance updated will be the minimum distance if SINGLE_LINKAGE, or maximum distance if COMPLETE_LINKAGE

        :param new_cluster: A set of data points to be merged as a new cluster
        """
        pairs_to_delete = []
        distance_lookup = {}
        for i, cluster_distances in enumerate(self.dist_adj_list):

            if len(cluster_distances[1].intersection(new_cluster)) > 0:  # TODO add comment here
                dist, pair = cluster_distances[0], cluster_distances[1]
                new_pair = frozenset(pair.union(new_cluster))

                if distance_lookup.has_key(new_pair):

                    if self.linkage == HierarchicalCluster.SINGLE_LINKAGE:
                        distance_lookup[new_pair] = np.min([dist, distance_lookup[new_pair]])
                    elif self.linkage == HierarchicalCluster.COMPLETE_LINKAGE:
                        distance_lookup[new_pair] = np.max([dist, distance_lookup[new_pair]])
                    elif self.linkage == HierarchicalCluster.AVERAGE_LINKAGE:
                        raise NotImplementedError()
                else:
                    distance_lookup[new_pair] = dist

                pairs_to_delete.append(pair)

        self.dist_adj_list = [x for x in self.dist_adj_list if not x[1].intersection(new_cluster)]

        for pair, distance in distance_lookup.iteritems():
            self.dist_adj_list.append((distance, pair))

        heapq.heapify(self.dist_adj_list)

    def euclidean_dist(self, point1, point2):
        return np.sqrt(np.sum(np.square(point1 - point2)))


def main():
    headers = ['height', 'weight', 'age']
    raw_data = [(170, 57, 32), (190, 95, 28), (150, 45, 35), (168, 65, 29), (175, 78, 26), (185, 90, 32),
                (171, 65, 28), (155, 48, 31), (165, 60, 27), (182, 80, 30), (175, 69, 28), (178, 80, 27),
                (160, 50, 31), (170, 72, 30)]
    dataset = pd.DataFrame(raw_data, columns=headers)
    # randdata = pd.DataFrame(np.random.randn(100, 10))

    hier_cluster = HierarchicalCluster()
    hier_cluster.train(dataset)


if __name__ == '__main__':
    main()
