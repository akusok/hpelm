# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np


def train_loo(self, X, T):
    """Model structure selection with Leave-One-Out (LOO) validation.

    Trains ELM, validates model with LOO and sets an optimal validated solution. Effect is similar to
    cross-validation with k==N, but ELM has explicit formula of solution for LOO without iterating k times.

    Args:
        self (ELM): ELM object that calls `train_v()`
        X (matrix): training set inputs
        T (matrix): training set outputs
    """

    H = self.nnet._project(X)
    self.add_data(X, T)
    HH, HT = self.nnet.get_corr()

    N = X.shape[0]
    L = self.nnet.L
    e = np.ones((L+1,)) * -1  # errors for all numbers of neurons
    rank, L = self._ranking(L, H, T)  # create ranking of neurons

    # PRESS_LOO
    P = np.linalg.inv(HH)
    Bt = np.dot(P, HT)
    R = np.ones((N,)) - np.einsum('ij,ji->i', np.dot(H, P), H.T)
    Y = np.dot(H, Bt)
    err = self._error(T, Y, R)

    penalty = err * 0.01 / L  # penalty is 1% of error at max(L)
    e[L] = err + L * penalty

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

    l = 1000  # run the while loop at least once
    while l > 2:
        # calculate errors at points
        for idx in [iA, iB, iC, iD, iE]:
            if e[idx] == -1:  # skip already calculated errors
                rank1 = rank[:idx]
                # H1 = H[:, rank1]
                H1 = np.take(H, rank1, 1)
                HH1 = HH[rank1, :][:, rank1]
                HT1 = HT[rank1, :]

                P = np.linalg.inv(HH1)
                Bt1 = np.dot(P, HT1)
                R = np.ones((N,)) - np.einsum('ij,ji->i', np.dot(H1, P), H1.T)
                Y1 = np.dot(H1, Bt1)
                err = self._error(T, Y1, R)
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

    self.nnet._prune(best_L)
    self.nnet.add_batch(X, T)
    self.nnet.solve()
#    print "%d of %d neurons selected with a LOO validation" % (len(best_nn), nn)
#    if len(best_nn) > nn*0.9:
#        print "Hint: try re-training with more hidden neurons"


































