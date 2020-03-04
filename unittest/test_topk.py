from __future__ import absolute_import

import unittest

from topk.topk import CPRow, CPTable, TopKOp  # noqa: F401
from topk.generate_test_case import gen_rand_cptable


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.num_row = 2
        self.count_max = 2
        self.k = 1

        self.cptable = gen_rand_cptable(self.num_row, self.count_max)

        # print table
        for cprow in self.cptable.data_:
            print('timestamp: {0:.3f}, prob: {1}'.format(cprow.timestamp(), cprow.prob()))


class TestCPTable(BaseTestCase):
    def setUp(self):
        super(TestCPTable, self).setUp()

    def test_get_top_conf(self):
        cptable = self.cptable

        for i in range(1, self.num_row+1):
            print(cptable.get_top_conf(range(i)))

        self.assertAlmostEqual(cptable.get_top_conf(range(self.num_row)), 1)

    def test_get_extended_top_conf(self):
        cptable = self.cptable

        for i in range(1, self.num_row+1):
            print(cptable.get_extended_top_conf(range(i), self.k))

        self.assertAlmostEqual(cptable.get_extended_top_conf(range(self.num_row), self.k), 1)


class TestTopKOp(BaseTestCase):
    def setUp(self):
        super(TestTopKOp, self).setUp()
        self.confidence = 0.5

    def test_forward(self):
        out_data = [None, None]
        op = TopKOp(table_path=None, k=self.k, confidence=self.confidence)
        op.forward(in_data=[self.cptable], out_data=out_data)
        print(out_data)


if __name__ == '__main__':
    unittest.main()
