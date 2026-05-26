import unittest

import numpy as np

import calc_drdv_and_sort as vsat


class PairCalculationTests(unittest.TestCase):
    def test_calc_dr_dv_without_errors_matches_known_pairs(self):
        n_stars = 3
        r = [[0.0, 1.0, 2.0]]
        v = [[0.0, 2.0, 4.0]]

        n_pairs, dr_dv, max_dr = vsat.calc_dr_dv(
            n_stars, r, v, error_flag=False, dv_default=True, bin_width=1.0
        )

        self.assertEqual(n_pairs, 3)
        np.testing.assert_allclose(
            np.sort(dr_dv[:, :2], axis=0),
            np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 4.0]]),
        )
        self.assertEqual(max_dr, 4.0)

    def test_calc_drdv_and_sort_returns_expected_bin_means(self):
        result = vsat.calc_drdv_and_sort(
            3,
            [[0.0, 1.0, 2.0]],
            [[0.0, 2.0, 4.0]],
            error_flag=False,
            dv_default=True,
            bin_width=1.0,
            min_tail_pairs=0,
        )

        n_bins, edges, mean_dv, error, n_in_bins, count_stars_bins = result

        self.assertEqual(n_bins, 2)
        np.testing.assert_allclose(edges, [0.0, 1.0])
        np.testing.assert_allclose(mean_dv, [2.0, 4.0])
        np.testing.assert_allclose(error, [0.0, 0.0])
        np.testing.assert_allclose(n_in_bins, [2.0, 1.0])
        np.testing.assert_array_equal(count_stars_bins[0], [1, 2, 1])
        np.testing.assert_array_equal(count_stars_bins[1], [1, 0, 1])

    def test_velocity_errors_produce_weighted_error(self):
        result = vsat.calc_drdv_and_sort(
            2,
            [[0.0, 1.0]],
            [[0.0, 2.0]],
            [[0.5, 0.5]],
            error_flag=True,
            dv_default=True,
            bin_width=1.0,
            min_tail_pairs=0,
        )

        _, _, mean_dv, error, n_in_bins, _ = result

        np.testing.assert_allclose(mean_dv, [2.0])
        np.testing.assert_allclose(error, [np.sqrt(0.5)])
        np.testing.assert_allclose(n_in_bins, [1.0])

    def test_radial_dv_is_signed(self):
        n_pairs, dr_dv, _ = vsat.calc_dr_dv(
            2,
            [[0.0, 1.0]],
            [[0.0, -2.0]],
            error_flag=False,
            dv_default=False,
            bin_width=1.0,
        )

        self.assertEqual(n_pairs, 1)
        np.testing.assert_allclose(dr_dv[0, :2], [1.0, -2.0])


if __name__ == "__main__":
    unittest.main()
