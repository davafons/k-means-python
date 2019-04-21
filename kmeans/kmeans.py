import time

from kmeans_engine import KMeansEngine
from kmeans_math import KMeansMath


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 10,
        tol: float = 1e-4,
        n_jobs: int = None,
    ):
        self.engine = KMeansEngine(n_clusters, n_init, max_iter, tol, n_jobs)

        if init == "random":
            self.use_random_centroids()
        else:
            self.use_kmeans_plusplus()

        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = None

    def fit(self, X):
        self.cluster_centers_, self.labels_, self.n_iter_ = self.engine.fit(X)

    def run_full_output(self, X):
        results = []

        start = time.clock()
        for (
            self.cluster_centers_,
            self.labels_,
            self.n_iter_,
        ) in self.engine.run_generator(X):
            end = time.clock() - start
            sse = KMeansMath.sse(X, self.cluster_centers_, self.labels_)

            row = (self.cluster_centers_, self.labels_, self.n_iter_, sse, end)
            results.append(row)

            start = time.clock()

        return results

    def use_euclidean_distance(self):
        self.engine.calc_distance = KMeansMath.euclidean_distance

    def use_random_centroids(self):
        self.engine.calc_initial_centroids = KMeansMath.random_centroids

    def use_kmeans_plusplus(self):
        self.engine.calc_initial_centroids = KMeansMath.kmeans_plusplus
