import math
from concurrent import futures

import numpy as np

from kmeans_math import KMeansMath


class KMeansEngine:
    """
    This class holds the actual implementation of the KMeans algorithm.
    """

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
        """
        Execute KMeans n times, where n is the value of 'n_init'. Return the cluster
        generated with the lowest SSE.
        """

        centroids = None
        labels = None
        i = None
        sse = math.inf

        # Iterate through each result of the KMeans execution
        for next_centroids, next_labels, next_i in self.fit_gen(X):
            next_sse = KMeansMath.sse(X, next_centroids, next_labels)

            # Save the result with lowest SSE
            if next_sse < sse:
                centroids, labels, i = next_centroids, next_labels, next_i
                sse = next_sse

        return centroids, labels, i

    def fit_gen(self, X):
        """
        Execute KMeans in parallel n times, where n is the value of 'n_init'. Yield the
        result of each execution.
        """

        # 'n_jobs' is the number of child process to create for parallel execution
        with futures.ProcessPoolExecutor(self.n_jobs_) as executor:
            result_futures = [
                executor.submit(self.run, X) for _ in range(0, self.n_init_)
            ]

            for future in futures.as_completed(result_futures):
                yield future.result()

    def run(self, X):
        """
        Run the KMeans once and return the last result (when clusters converge)
        """

        return [_ for _ in self.run_gen(X)][-1]

    def run_gen(self, X):
        """
        Run the KMeans and return the result of each step (Empty labels, first centroid
        recalculation, second centroid recalculation...) until the clusters converge or
        iterates 'max_iter_' times
        """

        # Initialize empty centroids and labels
        centroids = self.calc_initial_centroids(X, self.n_clusters_)
        labels = np.empty(shape=(X.shape[0],), dtype=int)

        # First, yield the KMeans before the algorithm starts
        yield centroids, np.zeros(shape=(X.shape[0],), dtype=int), 0

        for iteration in range(1, self.max_iter_):
            # Calculate the new centroids and new labels
            new_centroids, new_labels = self.__iter(X, centroids, labels)

            yield new_centroids, new_labels, iteration

            # Stop iterating if 'is_optimal' (convergence)
            if self.__is_optimal(centroids, new_centroids):
                break
            else:
                centroids, labels = new_centroids, new_labels

    def __iter(self, X, centroids, labels):
        """
        KMeans implementation. For each point, calculate the distance to the centroids
        and assign a label.
        Then, recalculate the centroids from the new assigned labels.
        """

        for i, point in enumerate(X):
            # Calculate the distance from the point to all the centroids
            distances = [self.calc_distance(point, centroid) for centroid in centroids]

            # The label is the index of the closest centroid
            label = distances.index(min(distances))
            # Update the labels array
            labels[i] = label

        # Recalculate the centroids with the new labels
        centroids = KMeansMath.recalculate_centroids(X, centroids, labels)

        return centroids, labels

    def __is_optimal(self, old_centroids, new_centroids):
        """
        For a cluster to be optimal, the difference between all the old and new
        centroids must be lower than the tolerance specified in the attribute 'tol'
        """

        for i in range(0, old_centroids.shape[0]):
            if np.sum(np.absolute(old_centroids[i] - new_centroids[i])) > self.tol_:
                return False

        return True
