# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np


def train_cv(self, X, T, k):
    N = X.shape[0]

    idxk = []
    for i in range(k):
        idxk.append(np.arange(N)[i::k])

    datak = []
    for i in range(k):
        items = [(i+j) % k for j in range(k)]
        idx_tr = np.hstack([idxk[j] for j in items[:-2]])
        idx_vl = idxk[items[-2]]
        idx_ts = idxk[items[-1]]
        Xtr = X[idx_tr]
        Ttr = T[idx_tr]
        Xvl = X[idx_vl]
        Tvl = T[idx_vl]
        Xts = X[idx_ts]
        Tts = T[idx_ts]
        HH, HT = self._project(Xtr, Ttr)
        Hvl = self.project(Xvl)
        Hts = self.project(Xts)
        rank, nn = self._ranking(Hvl.shape[1], Hvl, Tvl)
        datak.append((HH, HT, Hvl, Tvl, Hts, Tts, rank))

    e = np.ones((nn+1,)) * -1  # errors for all numbers of neurons

    err = 0
    for HH, HT, Hvl, Tvl, _, _, _ in datak:
        Beta = self._solve_corr(HH, HT)
        Yvl = np.dot(Hvl, Beta)
        err += self._error(Yvl, Tvl) / k
    penalty = err * 0.01 / nn  # penalty is 1% of error at max(nn)
    e[nn] = err + nn * penalty

    # MYOPT function
    # [A  B  C  D  E] interval points,
    # halve the interval each time

    # initialize intervals
    A = 3
    E = nn
    l = E - A
    B = A + l/4
    C = A + l/2
    D = A + 3*l/4

    l = 1000  # run the while loop at least once
    while l > 2:
        # calculate errors at points
        for idx in [A, B, C, D, E]:
            if e[idx] == -1:  # skip already calculated errors
                err = 0
                for HH, HT, Hvl, Tvl, _, _, rank in datak:
                    rank1 = rank[:idx]
                    HH1 = HH[rank1, :][:, rank1]
                    HT1 = HT[rank1, :]
                    Beta = self._solve_corr(HH1, HT1)
                    Yvl = np.dot(Hvl[:, rank1], Beta)
                    err += self._error(Yvl, Tvl) / k
                e[idx] = err + idx * penalty

        m = min(e[A], e[B], e[C], e[D], e[E])  # find minimum element

        # halve the search interval
        if m in (e[A], e[B]):
            E = C
            C = B
        elif m in (e[D], e[E]):
            A = C
            C = D
        else:
            A = B
            E = D
        l = E - A
        B = A + l/4
        D = A + (3*l)/4

    k_opt = [n1 for n1 in [A, B, C, D, E] if e[n1] == m][0]  # find minimum index
    best_nn = rank[:k_opt]

    # get test error
    err_ts = 0
    for HH, HT, _, _, Hts, Tts, _ in datak:
        Beta = self._solve_corr(HH, HT)
        Yts = np.dot(Hts, Beta)
        err_ts += self._error(Yts, Tts) / k

    self._prune(best_nn)
    self.Beta = self._project(X, T, solve=True)[2]
#    print "%d of %d neurons selected with a Cross-Validation" % (len(best_nn), nn)
#    print "the Cross-Validation test error is %f" % err_ts
#    if len(best_nn) > nn*0.9:
#        print "Hint: try re-training with more hidden neurons"
    return err_ts


















