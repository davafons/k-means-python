import numpy as np


class KMeansMath:
    @staticmethod
    def random_centroids(X, n_clusters):
        return np.array(KMeansMath.choose_points(X, n_clusters))

    @staticmethod
    def kmeans_plusplus(X, n_clusters):
        # Choose 1 random centroid first
        index = np.random.randint(0, X.shape[0], size=1)[0]
        centroids = np.array([X[index]])

        # Remove the picked centroid for avoiding repetitions
        X = np.delete(X, index, axis=0)

        # For the next k centroids, pick a random one using the squared distance from
        # the last centroid to the point as a probability weight
        for i in range(1, n_clusters):
            # Array of probabilities
            p = KMeansMath.sq_distance_prob_array(X, centroids[i - 1])

            # Pick the next centroid and remove
            index = np.random.choice(X.shape[0], 1, p=p)[0]
            centroids = np.append(centroids, [X[index]], axis=0)
            X = np.delete(X, index, axis=0)

        return centroids

    @staticmethod
    def choose_points(X, n_points):
        return np.array(
            [X[i] for i in np.random.choice(X.shape[0], n_points, replace=False)]
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
    def sq_distance_prob_array(X, centroid):
        distances = np.array([KMeansMath.sq_distance(point, centroid) for point in X])

        sum_distance = np.sum(distances)
        prob_distances = distances / sum_distance

        return prob_distances

    @staticmethod
    def euclidean_distance(A, B):
        return np.linalg.norm(A - B)

    @staticmethod
    def sq_distance(A, B):
        return KMeansMath.euclidean_distance(A, B) ** 2

    @staticmethod
    def sse(X, centroids, labels):
        return sum(
            [
                KMeansMath.sq_distance(point, centroids[labels[i]])
                for i, point in enumerate(X)
            ]
        )
