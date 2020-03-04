import unittest
from topk.topk import PQueue, MaxQueue


class TestPQueue(unittest.TestCase):
    def setUp(self):
        self.size = 5
        self.data = [2, 3, 1, 7, 5, 8, 6]

    def test_short(self):
        pq = PQueue(self.size, [3, 4])

        size_list = [3, 4, 5, 5, 5, 5, 5]
        min_list = [2, 2, 1, 2, 3, 3, 4]
        max_list = [4, 4, 4, 7, 7, 8, 8]
        for i in range(len(self.data)):
            v = self.data[i]
            pq.push(v)
            self.assertEqual(pq.size, size_list[i])
            self.assertEqual(pq.capacity, self.size)
            self.assertEqual(pq.min(), min_list[i])
            self.assertEqual(pq.max(), max_list[i])

    def test_long(self):
        pq = PQueue(self.size, [1, 4, 9, 1, 3, 2, 8])

        self.assertEqual(pq.size, self.size)
        self.assertEqual(pq.min(), 2)
        self.assertEqual(pq.max(), 9)

        pq.push(11)
        self.assertEqual(pq.min(), 3)
        self.assertEqual(pq.max(), 11)

    def test_init(self):
        init_list = sorted([3, 4, 5, 6, 7, 9], reverse=True)[:5]
        pq = PQueue(self.size, init_list)

        pq.push(11)
        self.assertEqual(pq.min(), 5)
        self.assertEqual(pq.max(), 11)

    def test_wo_capacity(self):
        pq = PQueue(0, [1, 2])

        size = len(pq)
        for v in self.data:
            pq.push(v)
            size += 1
            self.assertEqual(pq.size, size)


class TestMaxQueue(unittest.TestCase):
    def setUp(self):
        self.data = [3, 7, 9, 2, 4, 1, 4, 0]

    def test_maxqueue(self):
        mq = MaxQueue()

        for v in self.data:
            mq.push(v)

        max_list = [9, 9, 4, 4, 4, 4, 0]
        for i in range(len(mq)-1):
            mq.pop()
            self.assertEqual(mq.max(), max_list[i])


if __name__ == '__main__':
    unittest.main()
