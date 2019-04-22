import os
import re

import numpy as np
from sklearn import datasets


class InstanceLoader:
    """
    This class implements functions to load .txt or .csv files to a valid format (numpy
    arrays) for using with the KMeans algorithm.
    Additionally, some predefined datasets can be loaded (iris, blobs)
    """

    @staticmethod
    def load_dataset(name: str):
        """
        Recognize the extension or the name of the file passed and decide how to load
        the dataset
        """
        if name.lower() == "iris":
            return datasets.load_iris().data
        elif name.lower() == "blobs":
            X, _ = datasets.make_blobs()
            return X

        elif os.path.splitext(name)[-1].lower() == ".csv":
            return InstanceLoader.load_csv(name)

        return InstanceLoader.load_txt(name)

    @staticmethod
    def load_txt(filepath: str):
        """
        Load a dataset from a .txt file. The first two rows must be the number of rows
        and columns of the dataset, and the rest are the values to load.
        Asserts that the rows and values specified match the actual values loaded.
        """
        with open(filepath, "r") as input_file:
            expected_rows = int(input_file.readline())
            expected_cols = int(input_file.readline())

            data = []

            for row in input_file:
                values = [
                    float(value.replace(",", "."))
                    for value in re.split(r"\s+", row.strip())
                ]

                data.append(values)

            np_data = np.array(data)

            assert np_data.size == expected_cols * expected_rows

            return np_data

    @staticmethod
    def load_csv(filepath: str):
        """
        Load a dataset from a .csv file. The first row is skipped (header)
        """
        return np.genfromtxt(filepath, delimiter=";", skip_header=1)
