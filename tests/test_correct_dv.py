import unittest

import numpy as np

import correct_dv


class CorrectDvTests(unittest.TestCase):
    def test_average_pairwise_velocity_difference(self):
        average = correct_dv.average_pairwise_velocity_difference([[0.0, 2.0, 4.0]])
        self.assertEqual(average, 8.0 / 3.0)

    def test_draw_velocities_uses_pdf_range(self):
        draws = correct_dv.draw_velocities(
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            100,
            rng=123,
        )

        self.assertEqual(draws.shape, (100,))
        self.assertGreaterEqual(draws.min(), 0.0)
        self.assertLessEqual(draws.max(), 2.0)


if __name__ == "__main__":
    unittest.main()
