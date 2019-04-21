# import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans as KMeansSK

from instance_loader import InstanceLoader
from kmeans import KMeans


def main():
    X = InstanceLoader.load_txt("res/prob2.txt")
    # print(f"Instance loaded:\n{X}")

    # -- Print result for my KMeans implementation
    kmeans = KMeans(n_clusters=2)
    print("Personal KMeans implementation:")
    output_kmeans(kmeans, X)

    # -- Print result for sklearn KMeans implementation
    skmeans = KMeansSK(n_clusters=2)
    print("SKLearn KMeans implementation:")
    output_kmeans(skmeans, X)


def output_kmeans(kmeans, X):
    # Fit the model
    start = time.clock()
    kmeans.fit(X)
    end = time.clock() - start

    # Print result
    print(f"\tCentroids:\n{kmeans.cluster_centers_}")
    print(f"\tLabels:\n{kmeans.labels_}")
    print(f"\tNº iter: {kmeans.n_iter_}")
    print(f"\tExecution time: {end}")
    print("\n")


if __name__ == "__main__":
    main()
