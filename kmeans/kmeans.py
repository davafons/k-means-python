import numpy as np
from nptyping import Array


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 10,
        tol: float = 1e-4,
    ):
        self.n_clusters_: int = n_clusters
        self.n_init_: int = n_init
        self.max_iter_: int = max_iter
        self.tol_: float = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = None

    def fit(self, X: Array):
        self.cluster_centers_ = self.__initial_centroids(X)
        print("Initial centroids")
        print(self.cluster_centers_)

        for i in range(0, self.max_iter_):
            self.__run(X)
            print(self.labels_)
            self.n_iter_ = i

    def __run(self, X: Array):
        self.labels_ = np.empty(shape=(1, 0), dtype=int)

        for point in X:
            distances = [
                KMeans.__distance(point, centoroid)
                for centoroid in self.cluster_centers_
            ]

            label = distances.index(min(distances))
            self.labels_ = np.append(self.labels_, label)

        self.cluster_centers_ = self.__recalculate_centroids(X)

    def __initial_centroids(self, X: Array):
        return np.array(
            [
                X[i]
                for i in np.random.choice(
                    X.shape[0] - 1, self.n_clusters_, replace=False
                )
            ]
        )

    def __recalculate_centroids(self, X: Array):
        new_centroids = np.zeros(self.cluster_centers_.shape)

        # Calculate the sum of the points and the repetitions of each class
        repetitions = np.zeros((1, self.n_clusters_), dtype=int)
        for i in range(X.shape[0]):
            label = self.labels_[i]
            repetitions[:, label] += 1
            new_centroids[label] += X[i]

        # Divide each centroid by the number of points
        for i in range(new_centroids.shape[0]):
            new_centroids[i] /= repetitions[:, i]

        return new_centroids

    @staticmethod
    def __distance(A: Array, B: Array):
        return np.linalg.norm(A - B)
