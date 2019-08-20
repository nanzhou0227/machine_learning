import pandas as pd
import numpy as np
import random


def distance(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def find_centorid(points):
    x = np.mean([p.x for p in points])
    y = np.mean([p.y for p in points])
    return Point(x, y)


def initial_means(points, k):
    return random.sample(set(points), k)


def update(cluster, points):
    k = len(cluster)
    new_idxes = [p.find_close(cluster) for p in points]
    new_cluster = []
    for i in range(k):
        mask = np.array(new_idxes) == i
        new_cluster.append(find_centorid(list(np.array(points)[mask])))
    return new_cluster


def k_means(points, k):
    cluster = initial_means(points, k)
    while True:
        new_cluster = update(cluster, points)
        converge = True
        for i in range(k):
            if cluster[i] != new_cluster[i]:
                converge = False
                break
        if converge:
            break
        cluster = new_cluster

    return cluster


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "<{},{}>".format(self.x, self.y)

    def find_close(self, cluster):
        distance_list = [distance(self, p) for p in cluster]
        idx = np.argmin(distance_list)
        return idx

    def __eq__(self, other):
        return (self.x == other.x) & (self.y == other.y)

    def __hash__(self):
        return hash(self.__repr__())


if __name__ == "__main__":

    # Path of the file to read
    file_path = 'data/transactions.csv'
    location = pd.read_csv(file_path, header=None)
    location.columns = ["lat", "log"]
    points = []
    for _, row in location.iterrows():
        points.append(Point(row["lat"], row["log"]))

    k = 2
    result = k_means(points, k)
    print(result[0], result[1])




