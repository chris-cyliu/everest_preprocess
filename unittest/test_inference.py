from __future__ import absolute_import

import unittest
import numpy as np

from inference.inference import build_uncertain_table, build_uncertain_table_fast


class TestBuildUncertainTable(unittest.TestCase):
    def setUp(self):
        self.num_frames = 2
        self.max_bboxes = 3
        self.scores = np.random.rand(self.num_frames, self.max_bboxes)

    def test_build_uncertain_table(self):
        uncertain_table = build_uncertain_table(self.scores)

        # sum to 1
        np.testing.assert_array_almost_equal(np.sum(uncertain_table, axis=-1), 1)

        # compare with dummy method
        num_pw = 2 ** self.max_bboxes
        dummy_table = np.zeros((self.num_frames, self.max_bboxes + 1))

        for i in range(self.num_frames):
            for m in range(num_pw):
                num_objects = 0
                prob = 1

                for n in range(self.max_bboxes):
                    flag = m & (1 << n)
                    p = self.scores[i, n]

                    prob *= p if flag else 1 - p
                    num_objects += 1 if flag else 0

                dummy_table[i, num_objects] += prob

        np.testing.assert_array_almost_equal(uncertain_table, dummy_table)

    def test_build_uncertain_table_fast(self):
        uncertain_table = build_uncertain_table(self.scores)
        uncertain_table_fast = build_uncertain_table_fast(self.scores)

        # compare with exponential method
        np.testing.assert_array_almost_equal(uncertain_table, uncertain_table_fast)


if __name__ == '__main__':
    unittest.main()
