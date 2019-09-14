import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AverageNonzeroTripletsMetric(Metric):
    """
    Counts average number of nonzero triplets found in minibatches
    """

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


def average_precision(query, pred):
    match = 0
    temp = 0
    for i in range(len(pred)):
        if query == pred[i]:
            match += 1
            temp += match / (i+1)

    if match == 0:
        ap = 0
    else:
        ap = temp/match

    return ap


def cmc_score(rank_list, rank_max):
    num_rank = 0
    print("")
    for j in range(1, rank_max + 1):
        num_rank += rank_list.count(j)
        rank_value = (num_rank / len(rank_list)) * 100
        print("Rank {}: {}%".format(j, rank_value))


def open_set_scores(match_list, dist_list, test_lab, thresh=20):  # calcolo TTR e FTR
    tot = len(test_lab)
    non_target_tot = list(test_lab).count(0)
    target_tot = tot - non_target_tot
    target = 0
    non_target = 0
    if non_target_tot > 0:
        for z in range(tot):
            if dist_list[z] < thresh:
                if match_list[z] == test_lab[z]:
                    target += 1
                else:
                    if test_lab[z] == 0:
                        non_target += 1

        ttr = (target / target_tot) * 100
        ftr = (non_target / non_target_tot) * 100
        print("True target rate: {}%".format(ttr))
        print("False target rate: {}%".format(ftr))