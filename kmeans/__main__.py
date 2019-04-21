import argparse
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans as KMeansSK
from tabulate import tabulate

from instance_loader import InstanceLoader
from kmeans import KMeans


def main(args):
    X = InstanceLoader.load_dataset(args.dataset)

    if args.run_verbose:
        kmeans_run_verbose(X)

    else:
        compare_kmeans(X)


def kmeans_run_verbose(X):
    kmeans = KMeans(n_clusters=3, n_init=5, n_jobs=8)

    results = kmeans.run_full_output(X)

    headers = ["Centroids", "Labels", "Iteration", "SSE", "CPU"]
    print(tabulate(results, headers=headers))

    # for row in results:
    #     plot_kmeans(X, row[0], row[1], f"Iteration {row[2]}")
    kmeans_animation(X, results)

    plt.show()


def compare_kmeans(X):
    # -- Print result for my KMeans implementation
    kmeans = KMeans(n_clusters=3, n_init=100, n_jobs=8)

    print("Personal KMeans implementation:")
    output_kmeans(kmeans, X)
    plot_kmeans(
        plt.figure(),
        X,
        kmeans.cluster_centers_,
        kmeans.labels_,
        f"Personal KMeans implementation. Iterations before convergence: "
        f"{kmeans.n_iter_}",
    )

    # -- Print result for sklearn KMeans implementation
    skmeans = KMeansSK(n_clusters=3, n_init=100, n_jobs=8)
    print("SKLearn KMeans implementation:")
    output_kmeans(skmeans, X)
    plot_kmeans(
        plt.figure(),
        X,
        skmeans.cluster_centers_,
        skmeans.labels_,
        f"SKLearn KMeans implementation. Iterations before convergence: "
        f"{skmeans.n_iter_}",
    )

    plt.show()


def output_kmeans(kmeans, X):
    # Fit the model
    start = time.clock()
    kmeans.fit(X)
    end = time.clock() - start

    # Print result
    print(f"\tCentroids:\n{kmeans.cluster_centers_}")
    print(f"\tLabels:\n{kmeans.labels_}")
    print(f"\tNº iter: {kmeans.n_iter_}")
    print(f"\tExecution time: {end}")
    print("\n")


def plot_kmeans(fig, X, centroids, labels, title):
    fig.suptitle(title)

    dimension = X.shape[1]
    if dimension == 2:
        ax = fig.add_subplot(111)
        ax.scatter(X[:, 0], X[:, 1], c=labels)
        ax.scatter(centroids[:, 0], centroids[:, 1], s=130, marker="x")

    elif dimension >= 3:
        ax = fig.add_subplot(111, projection=Axes3D.name)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=130, marker="x")


def kmeans_animation(X, data):
    fig = plt.figure()

    def update_plot(i):
        fig.clear()
        plot_kmeans(fig, X, data[i][0], data[i][1], f"Iteration {i}")

    _ = animation.FuncAnimation(fig, update_plot, frames=len(data), repeat=True)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KMeans implementation, compared with SKLearn implementation"
    )
    parser.add_argument(
        "dataset",
        metavar="dataset",
        type=str,
        help="Path to the .txt file with the dataset to load, or one of the predefined "
        "datasets: (iris, blobs).",
    )
    parser.add_argument(
        "--run-verbose",
        help="Verbose output of only one KMeans run with the loaded dataset.",
        action="store_true",
    )

    main(parser.parse_args())
