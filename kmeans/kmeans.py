import time

from kmeans_engine import KMeansEngine
from kmeans_math import KMeansMath


class KMeans:
    """
    Interface for the KMeans implementation. Mimics some of the attributes and method
    from the SKLearn implementation.
    """

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
        """
        Return the result of executing the KMeans algorithm. Shows the generated cluster
        with the lowest SSE.
        """
        self.cluster_centers_, self.labels_, self.n_iter_ = self.engine.fit(X)

    def fit_full_output(self, X):
        """
        Return the output of all the KMeans clusters generated in parallel when calling
        fit.
        """
        return self.__full_output(X, self.engine.fit_gen)

    def run_full_output(self, X):
        """
        Return the output of all the intermediate steps generated when calling the
        KMeans method once.
        """
        return self.__full_output(X, self.engine.run_gen)

    def __full_output(self, X, gen_method):
        """
        Return an array with all the intermediate values from a generator function
        (run_gen or fit_gen), along the SSE and the CPU time taken.
        """
        results = []

        start = time.clock()
        for (self.cluster_centers_, self.labels_, self.n_iter_) in gen_method(X):
            end = time.clock() - start
            sse = KMeansMath.sse(X, self.cluster_centers_, self.labels_)

            row = (self.cluster_centers_, self.labels_, self.n_iter_, sse, end)
            results.append(row)

            start = time.clock()

        return results

    def use_euclidean_distance(self):
        """
        Use 'euclidean distance' for calculating the distance between two points.
        """
        self.engine.calc_distance = KMeansMath.euclidean_distance

    def use_random_centroids(self):
        """
        Use the 'random centroids' technique for generating the initial centroids.
        """
        self.engine.calc_initial_centroids = KMeansMath.random_centroids

    def use_kmeans_plusplus(self):
        """
        Use the 'kmeans++' technique for generating the initial centroids.
        """
        self.engine.calc_initial_centroids = KMeansMath.kmeans_plusplus
