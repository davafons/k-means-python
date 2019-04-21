import numpy as np


class KMeansMath:
    @staticmethod
    def random_centroids(X, n_clusters):
        return np.array(
            [X[i] for i in np.random.choice(X.shape[0] - 1, n_clusters, replace=False)]
        )

    @staticmethod
    def recalculate_centroids(X, centroids, labels):
        centroids = np.zeros(centroids.shape)
        repetitions = np.zeros((centroids.shape[0],), dtype=int)

        for i, label in enumerate(labels):
            repetitions[label] += 1
            centroids[label] += X[i]

        # Divide each centroid by the number of points
        for i in range(centroids.shape[0]):
            centroids[i] /= repetitions[i]

        return centroids

    @staticmethod
    def euclidean_distance(A, B):
        return np.linalg.norm(A - B)
