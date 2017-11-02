# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np


def train_v(self, X, T, Xv, Tv):
    """Model structure selection with a validation set.

    Trains ELM, validates model and sets an optimal validated solution.

    Args:
        self (ELM): ELM object that calls `train_v()`
        X (matrix): training set inputs
        T (matrix): training set outputs
        Xv (matrix): validation set inputs
        Tv (matrix): validation set outputs
    """
    self.add_data(X, T)
    HH, HT = self.nnet.get_corr()
    B = self.nnet.solve_corr(HH, HT)
    Hv = self.nnet._project(Xv)
    L = self.nnet.L
    e = np.ones((L+1,)) * -1  # errors for all numbers of neurons
    rank, L = self._ranking(L, Hv, Tv)  # create ranking of neurons

    Yv = np.dot(Hv, B)
    err = self._error(Tv, Yv)
    # TODO: replace penalty by Akaike-BIC criterion
    penalty = err*0.01 / L  # penalty is 1% of error at max(L)
    e[L] = err + L*penalty

    # MYOPT function
    # [iA  iB  iC  iD  iE] interval points,
    # halve the interval each time

    # initialize intervals
    iA = 1
    iE = L
    l = iE - iA
    iB = iA + l//4
    iC = iA + l//2
    iD = iA + 3*l//4

    l = 3  # run the while loop at least once
    while l > 2:
        # calculate errors at points
        for idx in [iA, iB, iC, iD, iE]:
            if e[idx] == -1:  # skip already calculated errors
                rank1 = rank[:idx]
                HH1 = HH[rank1, :][:, rank1]
                HT1 = HT[rank1, :]
                B = self.nnet.solve_corr(HH1, HT1)
                Yv = np.dot(Hv[:, rank1], B)
                e[idx] = self._error(Tv, Yv) + idx*penalty

        m = min(e[iA], e[iB], e[iC], e[iD], e[iE])  # find minimum element

        # halve the search interval
        if m in (e[iA], e[iB]):
            iE = iC
            iC = iB
        elif m in (e[iD], e[iE]):
            iA = iC
            iC = iD
        else:
            iA = iB
            iE = iD
        l = iE - iA
        iB = iA + l//4
        iD = iA + (3*l)//4

    k_opt = [n1 for n1 in [iA, iB, iC, iD, iE] if e[n1] == m][0]  # find minimum index
    best_L = rank[:k_opt]

    self.nnet._prune(best_L)
    self.nnet.add_batch(X, T)
    self.nnet.solve()
#    print "%d of %d neurons selected with a validation set" % (len(best_nn), nn)
#    if len(best_nn) > nn*0.9:
#        print "Hint: try re-training with more hidden neurons"


















