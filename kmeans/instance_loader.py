import re

import numpy as np
from nptyping import Array


class InstanceLoader:
    @staticmethod
    def load_txt(filepath: str) -> Array:
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
