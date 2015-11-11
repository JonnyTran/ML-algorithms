import numpy as np
import pandas as pd


class HierarchicalCluster:
    SINGLE_LINKAGE = 0
    COMPLETE_LINKAGE = 1

    def __init__(self, linkage=SINGLE_LINKAGE):
        pass

    def merge(self):
        pass

    def distance(self, point1, point2):
        return np.sqrt(np.sum(np.square(point1 - point2)))


def main():
    headers = ['height', 'weight', 'age']
    raw_data = [(170, 57, 32), (190, 95, 28), (150, 45, 35), (168, 65, 29), (175, 78, 26), (185, 90, 32),
                (171, 65, 28), (155, 48, 31), (165, 60, 27), (182, 80, 30), (175, 69, 28), (178, 80, 27),
                (160, 50, 31), (170, 72, 30)]
    dataset = pd.DataFrame(raw_data, columns=headers)
    print dataset


if __name__ == '__main__':
    main()
