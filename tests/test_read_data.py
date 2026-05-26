import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import read_data


class ReadDataTests(unittest.TestCase):
    def test_reads_gaia_style_dataframe(self):
        df = pd.DataFrame(
            {
                "ra": [10.0, 11.0, 12.0],
                "dec": [-2.0, -1.5, -1.0],
                "pmra": [1.0, 2.0, 3.0],
                "pmdec": [4.0, 5.0, 6.0],
                "pmra_error": [0.1, 0.1, 0.2],
                "pmdec_error": [0.2, 0.2, 0.3],
            }
        )

        n_stars, r, v, verr = read_data.read_data(df)

        self.assertEqual(n_stars, 3)
        np.testing.assert_allclose(r, [[10.0, 11.0, 12.0], [-2.0, -1.5, -1.0]])
        np.testing.assert_allclose(v, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(verr, [[0.1, 0.1, 0.2], [0.2, 0.2, 0.3]])

    def test_reads_csv_with_explicit_columns_and_drops_missing_rows(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, np.nan],
                "y": [0.0, 1.0, 2.0],
                "vx": [2.0, 3.0, 4.0],
                "vy": [5.0, 6.0, 7.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.csv"
            df.to_csv(path, index=False)
            n_stars, r, v, verr = read_data.read_data(
                path,
                position_cols=["x", "y"],
                velocity_cols=["vx", "vy"],
            )

        self.assertEqual(n_stars, 2)
        np.testing.assert_allclose(r, [[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(v, [[2.0, 3.0], [5.0, 6.0]])
        self.assertEqual(verr, 0)


if __name__ == "__main__":
    unittest.main()
