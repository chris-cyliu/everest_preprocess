from topk import CPRow, CPTable
from itertools import combinations

def topk_prob(table, k, candidates, break_tie=True):
    if len(table) < k:
        return 0
    if len(candidates) < k:
        return 0

    total_prob = 0
    for pw, prob in table.iter_possible_worlds():
        pw.sort()
        topk = set([pw[i].timestamp() for i in range(k)])

        if not break_tie:
            for i in range(k, len(table)):
                if pw[k].count != pw[i].count:
                    break
                topk.add(pw[i].timestamp())

        if topk.issuperset(candidates):
            total_prob += prob

    return total_prob
    
def most_prob_topk(table, k, break_tie=True):
    timestamps = [row.timestamp() for row in table]
    
    max_prob = 0
    max_prob_candidate = []
    for candidate in combinations(timestamps, k):
        prob = topk_prob(table, k, candidate, break_tie)
        if prob > max_prob:
            max_prob = prob
            max_prob_candidate = candidate
    return max_prob_candidate, max_prob