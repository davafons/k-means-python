from kmeans_engine import KMeansEngine
from kmeans_math import KMeansMath


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 10,
        tol: float = 1e-4,
    ):
        self.engine = KMeansEngine(n_clusters, n_init, max_iter, tol)
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = None

    def fit(self, X):
        self.cluster_centers_, self.labels_, self.n_iter_ = self.engine.fit(X)

    def use_euclidean_distance(self):
        self.engine.calc_distance = KMeansMath.euclidean_distance

    def use_random_centroids(self):
        self.engine.calc_initial_centroids = KMeansMath.random_centroids
