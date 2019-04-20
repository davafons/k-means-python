import re

import pandas as pd


class InstanceLoader:
    @staticmethod
    def load_txt(filepath: str) -> pd.DataFrame:
        with open(filepath, "r") as input_file:
            expected_rows = int(input_file.readline())
            expected_cols = int(input_file.readline())

            data = []

            for row in input_file:
                values = [
                    float(value.replace(",", "."))
                    for value in re.split(r"\s+", row.strip())
                ]

                InstanceLoader.__assert_len(values, expected_cols)

                data.append(values)

            InstanceLoader.__assert_len(data, expected_rows)

            return pd.DataFrame(data)

    @staticmethod
    def __assert_len(array, expected_len: int):
        if len(array) != expected_len:
            raise SyntaxError(
                f"{__name__}: Expected {expected_len} "
                f"values on {array[:5]}... Found {len(array)}"
            )
