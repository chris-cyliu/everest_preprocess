from topk import CPTable, CPRow
from brute_force import topk_prob, most_prob_topk
from itertools import combinations
import unittest

class TestBruteForce(unittest.TestCase):
    def test_enumerate_pw(self):
        table_data = [
            CPRow(0, [0, 0.3, 0.4, 0.3]),
            CPRow(1, [0, 0.2, 0.5, 0.3]),
            CPRow(2, [0, 0.2, 0.5, 0.3]),
        ]
        table = CPTable(table_data)
        pws = table.iter_possible_worlds()
        total_prob = 0
        count = 0
        for pw, prob in pws:
            total_prob += prob
            count += 1
        self.assertTrue(abs(total_prob - 1) < 0.0001)
        self.assertEqual(count, len(table_data)**3)
    
    def test_topk_prob(self):
        table_data = [
            CPRow(0, [0, 0.3, 0.4, 0.3]),
            CPRow(1, [0, 0.2, 0.5, 0.3]),
            CPRow(2, [0, 0.2, 0.4, 0.4]),
            CPRow(3, [0, 0.2, 0.4, 0.4]),
            CPRow(4, [0, 0.2, 0.4, 0.4]),
            CPRow(5, [0, 0.2, 0.4, 0.4]),
        ]
        table = CPTable(table_data)
        total_prob = 0
        for sub in combinations([0, 1, 2, 3, 4, 5], 2):
            prob = topk_prob(table, 2, sub)
            #print(sub, "prob:", prob)
            total_prob += prob
        self.assertTrue(abs(total_prob - 1) < 0.0001)

    def test_topk_prob_with_tie(self):
        table_data = [
            CPRow(0, [0, 0.3, 0.4, 0.3]),
            CPRow(1, [0, 0.2, 0.5, 0.3]),
            CPRow(2, [0, 0.2, 0.4, 0.4]),
            CPRow(3, [0, 0.2, 0.4, 0.4]),
            CPRow(4, [0, 0.2, 0.4, 0.4]),
            CPRow(5, [0, 0.2, 0.4, 0.4]),
        ]
        table = CPTable(table_data)
        total_prob = 0
        for sub in combinations([0, 1, 2, 3, 4, 5], 2):
            prob = topk_prob(table, 2, sub, False)
            #print(sub, "prob:", prob)
            total_prob += prob
        # something interesting happened here
        self.assertTrue(abs(total_prob - 3) < 0.0001)

    def test_most_prob_topk(self):
        table_data = [
            CPRow(0, [0, 0.3, 0.3, 0.4]),
            CPRow(1, [0, 0.2, 0.5, 0.3]),
            CPRow(2, [0, 0.2, 0.4, 0.4]),
            CPRow(3, [0, 0.2, 0.4, 0.4]),
            CPRow(4, [0, 0.2, 0.4, 0.4]),
        ]
        table = CPTable(table_data)
        result, _ = most_prob_topk(table, 2)
        self.assertEqual(result, (3, 4))

if __name__ == '__main__':
    unittest.main()
