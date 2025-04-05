import random
import numpy as np

def calc_probability(scores, genome=True):
    if genome:
        scores = np.array([i.score for i in scores], dtype=np.float64)
    else:
        scores = np.array(scores, dtype=np.float64)
    min_score = np.min(scores)
    add = 1e-14
    if min_score < 0:
        add -= min_score
    scores += add
    sum_scores = np.sum(scores)
    probabilities = np.cumsum(scores / sum_scores)
    probabilities = np.insert(probabilities, 0, 0)
    probabilities[-1] = 1
    return probabilities.tolist()

def select(probabilities):
    value = random.uniform(0, 1)
    for i in range(1, len(probabilities)):
        if value < probabilities[i]:
            return i - 1
    return len(probabilities) - 2
