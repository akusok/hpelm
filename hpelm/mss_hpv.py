# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np


def train_hpv(self, HH, HT, Xv, Tv, Yv):
    Beta = self._
    
    HH, HT, Beta = self._project(X, T, solve=True)
    Hv = self.project(Xv)
    nn = Hv.shape[1]
    e = np.ones((nn+1,)) * -1  # errors for all numbers of neurons
    rank, nn = self._ranking(nn, Xv, Tv)  # create ranking of neurons

    Yv = np.dot(Hv, Beta)
    err = self._error(Yv, Tv)
    penalty = err * 0.01 / nn  # penalty is 1% of error at max(nn)
    e[nn] = err + nn * penalty

    # MYOPT function
    # [A  B  C  D  E] interval points,
    # halve the interval each time

    # initialize intervals
    A = 1
    E = nn
    l = E - A
    B = A + l/4
    C = A + l/2
    D = A + 3*l/4

    l = 3  # run the while loop at least once
    while l > 2:
        # calculate errors at points
        for idx in [A, B, C, D, E]:
            if e[idx] == -1:  # skip already calculated errors
                rank1 = rank[:idx]
                HH1 = HH[rank1, :][:, rank1]
                HT1 = HT[rank1, :]
                Beta = self._solve_corr(HH1, HT1)
                Yv = np.dot(Hv[:, rank1], Beta)
                e[idx] = self._error(Yv, Tv) + idx * penalty

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

    self._prune(best_nn)
    self.Beta = self._project(X, T, solve=True)[2]
    print "%d of %d neurons selected with a validation set" % (len(best_nn), nn)
    if len(best_nn) > nn*0.9:
        print "Hint: try re-training with more hidden neurons"


















