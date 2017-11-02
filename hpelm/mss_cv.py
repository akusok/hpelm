# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np


def train_cv(self, X, T, k):
    """Model structure selection with cross-validation.

    Trains ELM, cross-validates model and sets an optimal validated solution.

    Args:
        self (ELM): ELM object that calls `train_v()`
        X (matrix): training set inputs
        T (matrix): training set outputs
        k (int): number of parts to split the dataset into, k-2 parts are used for training and 2 parts are
            left out: 1 for validation and 1 for test; repeated k times until all parts have been left out for
            validation and test, and results averaged over these k repetitions.

    Returns:
        err_t (double): error for the optimal model, computed in the 'cross-testing' manner on data part
        which is not used for training or validation
    """
    N = X.shape[0]
    L = self.nnet.L
    c = self.nnet.outputs

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
        self.nnet.reset()
        self.nnet.add_batch(Xtr, Ttr)
        HH, HT = self.nnet.get_corr()
        Hvl = self.nnet._project(Xvl)
        Hts = self.nnet._project(Xts)
        rank, L = self._ranking(Hvl.shape[1], Hvl, Tvl)
        datak.append((HH, HT, Hvl, Tvl, Hts, Tts, rank))

    e = np.ones((L+1,)) * -1  # errors for all numbers of neurons

    err = 0
    for HH, HT, Hvl, Tvl, _, _, _ in datak:
        B = self.nnet.solve_corr(HH, HT)
        Yvl = np.dot(Hvl, B)
        err += self._error(Tvl, Yvl) / k
    penalty = err * 0.01 / L  # penalty is 1% of error at max(L)
    e[L] = err + L * penalty

    # MYOPT function
    # [iA  iB  iC  iD  iE] interval points,
    # halve the interval each time

    # initialize intervals
    iA = 3
    iE = L
    l = iE - iA
    iB = iA + l//4
    iC = iA + l//2
    iD = iA + 3*l//4

    # TODO: tell about single-letter matrix notations in the whole code
    l = 1000  # run the while loop at least once
    while l > 2:
        # calculate errors at points
        for idx in [iA, iB, iC, iD, iE]:
            if e[idx] == -1:  # skip already calculated errors
                err = 0
                for HH, HT, Hvl, Tvl, _, _, rank in datak:
                    rank1 = rank[:idx]
                    HH1 = HH[rank1, :][:, rank1]
                    HT1 = HT[rank1, :]
                    B = self.nnet.solve_corr(HH1, HT1)
                    Yvl = np.dot(Hvl[:, rank1], B)
                    err += self._error(Tvl, Yvl) / k
                e[idx] = err + idx * penalty

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

    # get test error
    err_t = 0
    for HH, HT, _, _, Hts, Tts, _ in datak:
        B = self.nnet.solve_corr(HH, HT)
        Yts = np.dot(Hts, B)
        err_t += self._error(Tts, Yts) / k

    self.nnet._prune(best_L)
    self.nnet.add_batch(X, T)
    self.nnet.solve()
#    print "%d of %d neurons selected with a Cross-Validation" % (len(best_nn), nn)
#    print "the Cross-Validation test error is %f" % err_ts
#    if len(best_nn) > nn*0.9:
#        print "Hint: try re-training with more hidden neurons"
    return err_t


















