import unittest
from topk.topk import GSList, VT


class TestGSList(unittest.TestCase):
    def setUp(self):
        self.k = 4
        self.gap = 3

    def test_no_overlap(self):
        data = [VT(v=4, t=1), VT(v=3, t=5), VT(v=7, t=11), VT(v=1, t=15), VT(v=8, t=19), VT(v=7, t=23)]
        gslist = GSList(self.gap, self.k)

        min_list = [4, 3, 3, 1, 3, 4]
        max_list = [4, 4, 7, 7, 8, 8]
        for i in range(len(data)):
            v = data[i]
            gslist.push(v)
            self.assertEqual(gslist.min(1).v, min_list[i])
            self.assertEqual(gslist.max(1).v, max_list[i])

    def test_overlap(self):
        data = [VT(v=7, t=1), VT(v=3, t=5), VT(v=4, t=11), VT(v=7, t=15), VT(v=8, t=10), VT(v=7, t=9), VT(v=6, t=6)]
        gslist = GSList(self.gap, self.k)

        min_list = [7, 3, 3, 3, 3, 3, 6]
        max_list = [7, 7, 7, 7, 8, 8, 8]
        for i in range(len(data)):
            v = data[i]
            gslist.push(v)
            self.assertEqual(gslist.min(1).v, min_list[i])
            self.assertEqual(gslist.max(1).v, max_list[i])


if __name__ == '__main__':
    unittest.main()
