import matplotlib.pyplot as plt
from nptyping import Array

from instance_loader import InstanceLoader
from kmeans import KMeans

# from sklearn.cluster import KMeans


def main():
    X: Array = InstanceLoader.load_txt("res/prob3.txt")
    # print(f"Instance loaded:\n{X}")

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=130, marker="x"
    )
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.show()


if __name__ == "__main__":
    main()
