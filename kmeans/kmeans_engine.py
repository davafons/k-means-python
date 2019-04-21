import math
from concurrent import futures

import numpy as np

from kmeans_math import KMeansMath


class KMeansEngine:
    def __init__(
        self, n_clusters: int, n_init: int, max_iter: int, tol: float, n_jobs: int
    ):
        self.n_clusters_: int = n_clusters
        self.n_init_: int = n_init
        self.max_iter_: int = max_iter
        self.tol_: float = tol
        self.n_jobs_: int = n_jobs

        # Default method to find the distance between two points
        self.calc_distance = KMeansMath.euclidean_distance

        # Default method to set the initial centroids
        self.calc_initial_centroids = KMeansMath.kmeans_plusplus

    def fit(self, X):
        centroids = None
        labels = None
        i = None
        sse = math.inf

        with futures.ProcessPoolExecutor(self.n_jobs_) as executor:
            result_futures = [
                executor.submit(self.run, X) for _ in range(0, self.n_init_)
            ]

            for future in futures.as_completed(result_futures):
                new_centroids, new_labels, new_i = future.result()
                new_sse = KMeansMath.sse(X, new_centroids, new_labels)

                if new_sse < sse:
                    centroids, labels, i = new_centroids, new_labels, new_i
                    sse = new_sse

        return centroids, labels, i

    def run(self, X):
        return [_ for _ in self.run_generator(X)][-1]

    def run_generator(self, X):
        # Initialize empty centroids and labels
        centroids = self.calc_initial_centroids(X, self.n_clusters_)
        labels = np.empty(shape=(X.shape[0],), dtype=int)

        yield centroids, np.zeros(shape=(X.shape[0],), dtype=int), 0

        for iteration in range(1, self.max_iter_):
            new_centroids, new_labels = self.__iter(X, centroids, labels)
            yield new_centroids, new_labels, iteration

            if self.__is_optimal(centroids, new_centroids):
                break
            else:
                centroids, labels = new_centroids, new_labels

    def __iter(self, X, centroids, labels):
        for i, point in enumerate(X):
            # Calculate the distance from the point to all the centroids
            distances = [self.calc_distance(point, centroid) for centroid in centroids]

            # The label is the index of the closest centroid
            label = distances.index(min(distances))
            # Update the labels array
            labels[i] = label

        centroids = KMeansMath.recalculate_centroids(X, centroids, labels)

        return centroids, labels

    def __is_optimal(self, old_centroids, new_centroids):
        for i in range(0, old_centroids.shape[0]):
            if np.sum(np.absolute(old_centroids[i] - new_centroids[i])) > self.tol_:
                return False

        return True
