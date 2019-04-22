import numpy as np


class KMeansMath:
    """
    Mathematical operations used by the KMeans (Euclidean distance, Centroids
    generation, SSE, random choice...)
    """

    @staticmethod
    def random_centroids(X, n_clusters):
        """
        For an array of X points, return 'n_clusters' different points without
        repetition.
        """

        return np.array(KMeansMath.choose_points(X, n_clusters))

    @staticmethod
    def kmeans_plusplus(X, n_clusters):
        """
        Implementation of the 'kmeans++' algorithm for selecting the initial centroids.
        First, chose the first centroid randomly.
        Then, calculate an array with the squared distance from the centroid to any
        other point. Each distance will be the probability of picking this point as the
        next centroid. Transform the array of distances to an array of probabilities.
        Select the next 'n_clusters - 1' centroids using this method.
        """

        # Choose 1 random centroid first
        index = np.random.RandomState().randint(0, X.shape[0], size=1)[0]
        centroids = np.array([X[index]])

        # Remove the picked centroid for avoiding repetition
        X = np.delete(X, index, axis=0)

        # For the next k centroids, pick a random one using the squared distance from
        # the last centroid to the point as a probability weight
        for i in range(1, n_clusters):
            # Array of probabilities
            p = KMeansMath.sq_distance_prob_array(X, centroids[i - 1])

            # Pick the next centroid using the probabilities and remove it
            index = np.random.RandomState().choice(X.shape[0], 1, p=p)[0]
            centroids = np.append(centroids, [X[index]], axis=0)
            X = np.delete(X, index, axis=0)

        return centroids

    @staticmethod
    def choose_points(X, n_points):
        """
        From an array of X points, choose 'n_points' randomly without repetition.
        """
        return np.array(
            [
                X[i]
                for i in np.random.RandomState().choice(
                    X.shape[0], n_points, replace=False
                )
            ]
        )

    @staticmethod
    def recalculate_centroids(X, centroids, labels):
        """
        Recalculate the centroids from the new array of labels. The new centroid is the
        mean point from all the labels of the same class.
        """

        # New centroids
        centroids = np.zeros(centroids.shape)
        repetitions = np.zeros((centroids.shape[0],), dtype=int)

        # Sum the points with equal classes
        for i, label in enumerate(labels):
            repetitions[label] += 1
            centroids[label] += X[i]

        # Divide each centroid by the number of points that each class has
        for i in range(centroids.shape[0]):
            centroids[i] /= repetitions[i]

        return centroids

    @staticmethod
    def sq_distance_prob_array(X, centroid):
        """
        Return an array of probabilities calculated from the squared distances from a
        centroid to the rest of points of X.
        """

        # Squared distances from centroid to the X points.
        distances = np.array([KMeansMath.sq_distance(point, centroid) for point in X])

        # Transform the distances to probabilities (must sum up to 1.0)
        sum_distance = np.sum(distances)
        prob_distances = distances / sum_distance

        return prob_distances

    @staticmethod
    def euclidean_distance(A, B):
        """
        Return the euclidean distance from two point A and B, using the 'numpy'
        implementation.
        """

        return np.linalg.norm(A - B)

    @staticmethod
    def sq_distance(A, B):
        """
        Return the euclidean distance from two points A and B, squared.
        """

        return KMeansMath.euclidean_distance(A, B) ** 2

    @staticmethod
    def sse(X, centroids, labels):
        """
        Return the SSE. The SSE is calculated as the sum of all the squared distances
        from a point to its assigned centroid, for all points.
        """

        return sum(
            [
                KMeansMath.sq_distance(point, centroids[labels[i]])
                for i, point in enumerate(X)
            ]
        )
