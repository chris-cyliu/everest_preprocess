import unittest
from topk.topk import PQueue, VT, SEV, point_to_slab


class TestSlab(unittest.TestCase):
    def setUp(self):
        self.gap = 3

    def test_bound(self):
        data = PQueue(
            2,
            [VT(v=18, t=3), VT(v=11, t=17)]
        )
        data = data.sorted()

        slabs = point_to_slab(data, self.gap, 0, 20)
        self.assertEqual(slabs, [SEV(s=0, e=6, v=18), SEV(s=7, e=20, v=11)])

        slabs = point_to_slab(data, self.gap, -5, 21)
        self.assertEqual(slabs, [SEV(s=-5, e=-1, v=11), SEV(s=0, e=6, v=18), SEV(s=7, e=21, v=11)])

        slabs = point_to_slab(data, self.gap, 1, 19)
        self.assertEqual(slabs, [SEV(s=1, e=6, v=18), SEV(s=7, e=19, v=11)])

    def test_point_overlap(self):
        data = PQueue(
            2,
            [VT(v=1, t=3), VT(v=8, t=9)]
        )
        data = data.sorted()

        slabs = point_to_slab(data, self.gap, 0, 20)
        self.assertEqual(slabs, [SEV(s=0, e=5, v=1), SEV(s=6, e=12, v=8), SEV(s=13, e=20, v=1)])

    def test_interval_overlap(self):
        data = PQueue(
            2,
            [VT(v=1, t=3), VT(v=8, t=5)]
        )
        data = data.sorted()

        slabs = point_to_slab(data, self.gap, 0, 20)
        self.assertEqual(slabs, [SEV(s=0, e=1, v=1), SEV(s=2, e=8, v=8), SEV(s=9, e=20, v=1)])

        data = PQueue(
            5,
            [VT(v=9, t=3), VT(v=11, t=8), VT(v=8, t=9), VT(v=3, t=15), VT(v=18, t=17)]
        )
        data = data.sorted()

        slabs = point_to_slab(data, self.gap, 0, 20)
        self.assertEqual(slabs, [SEV(s=0, e=4, v=9), SEV(s=5, e=11, v=11), SEV(s=12, e=12, v=8), SEV(s=13, e=13, v=3), SEV(s=14, e=20, v=18)])

    def test_empty(self):
        data = []
        slabs = point_to_slab(data, self.gap, 0, 20)
        self.assertEqual(slabs, [SEV(s=0, e=20, v=0)])


if __name__ == '__main__':
    unittest.main()
