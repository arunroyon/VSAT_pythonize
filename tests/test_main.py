import unittest

import pandas as pd

import main


class AnalyzeTests(unittest.TestCase):
    def test_correct_inflation_rejects_directional_dv(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "vx": [0.0, 1.0, 2.0],
                "vx_error": [0.1, 0.1, 0.1],
            }
        )

        with self.assertRaisesRegex(ValueError, "dv_default=True"):
            main.analyze(
                df,
                position_cols=["x"],
                velocity_cols=["vx"],
                velocity_error_cols=["vx_error"],
                dv_default=False,
                correct_inflation=True,
            )

    def test_correct_inflation_rejects_heterogeneous_uncertainties(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "vx": [0.0, 1.0, 2.0],
                "vx_error": [0.1, 0.2, 0.3],
            }
        )

        with self.assertRaisesRegex(ValueError, "uniform velocity uncertainties"):
            main.analyze(
                df,
                position_cols=["x"],
                velocity_cols=["vx"],
                velocity_error_cols=["vx_error"],
                dv_default=True,
                correct_inflation=True,
                min_tail_pairs=0,
            )

    def test_analyze_exposes_bin_centers(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "vx": [0.0, 2.0, 4.0],
            }
        )

        result = main.analyze(
            df,
            position_cols=["x"],
            velocity_cols=["vx"],
            error_flag=False,
            bin_width=1.0,
            min_tail_pairs=0,
        )

        self.assertEqual(result.bin_width, 1.0)
        self.assertEqual(result.bin_centers.tolist(), [0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
