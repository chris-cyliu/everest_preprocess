# generate test case for top-k operator
from __future__ import absolute_import

import argparse
import csv
import random
import numpy as np

from .topk import CPRow, CPTable


def gen_rand_prob(size):
    prob = [random.random() for i in range(size)]
    s = np.sum(prob)
    prob /= s
    return prob


def gen_rand_cprow(size):
    timestamp = random.random()
    prob = gen_rand_prob(size)
    cprow = CPRow(timestamp, prob)
    return cprow


def gen_rand_cptable(num_row, count_max, save_path=None):
    cprow_list = [gen_rand_cprow(count_max) for _ in range(num_row)]
    cptable = CPTable(cprow_list)

    if save_path is not None:
        with open(save_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'prob']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for cprow in cptable:
                writer.writerow({
                    'timestamp': cprow.timestamp(),
                    'prob': cprow.prob()
                })

    return cptable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test cases for Top-K.')
    parser.add_argument('--num_row', type=int, help='Number of rows in table')
    parser.add_argument('--count_max', type=int, help='Number of columns in table')
    parser.add_argument('--save_path', type=str, help='Path for saving to csv file')

    args = parser.parse_args()

    gen_rand_cptable(args.num_row, args.count_max, args.save_path)
