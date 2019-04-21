import numpy as np
from nptyping import Array


class KMeansEngine:
    def __init__(self, n_clusters: int, n_init: int, max_iter: int, tol: float):
        self.n_clusters_: int = n_clusters
        self.n_init_: int = n_init
        self.max_iter_: int = max_iter
        self.tol_: float = tol

    def run(self, X: Array) -> (Array, Array):
        """
        Yield the centroids, labels and nº iteration
        """
        centroids = self.__initial_centroids(X)
        labels = np.empty(shape=(1, X.shape[1]), dtype=int)

        for i in range(0, self.max_iter_):
            # Execute the next iteration
            KMeansEngine.__iter(X, centroids, labels)

            # Yield the results
            yield centroids, labels, i

    def __initial_centroids(self, X: Array):
        return np.array(
            [
                X[i]
                for i in np.random.choice(
                    X.shape[0] - 1, self.n_clusters_, replace=False
                )
            ]
        )

    @staticmethod
    def __iter(X: Array, centroids: Array, labels: Array):
        for i, point in enumerate(X):
            # Calculate the distance from the point to all the centroids
            distances = [
                KMeansEngine.__distance(point, centroid) for centroid in centroids
            ]

            # The label is the index of the closest centroid
            label = distances.index(min(distances))
            # Update the labels array
            labels[X:, i] = label

        KMeansEngine.__recalculate_centroids(X, centroids, labels)

    @staticmethod
    def __recalculate_centroids(X: Array, centroids: Array, labels: Array):
        centroids = np.zeros(centroids.shape)
        repetitions = np.zeros((1, centroids.shape[0]), dtype=int)

        for i, label in enumerate(labels):
            repetitions[:, label] += 1
            centroids[label] += X[i]

        # Divide each centroid by the number of points
        for i in range(centroids.shape[0]):
            centroids[i] /= repetitions[:, i]

    @staticmethod
    def __distance(A: Array, B: Array):
        return np.linalg.norm(A - B)
