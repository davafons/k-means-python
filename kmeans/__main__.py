import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from instance_loader import InstanceLoader


def main():
    df = InstanceLoader.load_txt("res/prob1.txt")
    print(f"DataFrame loaded:\n{df.head()}")

    # Test: Plot the dataframe
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[0], df[1], df[2], marker="s")

    plt.show()


if __name__ == "__main__":
    main()
