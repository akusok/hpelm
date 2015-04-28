# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import inv


def train_loo(self, X, T):
    H = self.project(X)
    HH, HT = self._project(X, T)

    N, nn = H.shape
    e = np.ones((nn+1,)) * -1  # errors for all numbers of neurons
    rank, nn = self._ranking(nn, H, T)  # create ranking of neurons

    # PRESS_LOO
    P = inv(HH)
    Bt = np.dot(P, HT)
    R = np.ones((N,)) - np.einsum('ij,ji->i', np.dot(H, P), H.T)
    Y = np.dot(H, Bt)
    err = self._error(Y, T, R)

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

    l = 1000  # run the while loop at least once
    while l > 2:
        # calculate errors at points
        for idx in [A, B, C, D, E]:
            if e[idx] == -1:  # skip already calculated errors
                rank1 = rank[:idx]
                # H1 = H[:, rank1]
                H1 = np.take(H, rank1, 1)
                HH1 = HH[rank1, :][:, rank1]
                HT1 = HT[rank1, :]

                P = inv(HH1)
                Bt1 = np.dot(P, HT1)
                R = np.ones((N,)) - np.einsum('ij,ji->i', np.dot(H1, P), H1.T)
                Y1 = np.dot(H1, Bt1)
                err = self._error(Y1, T, R)
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

    self._prune(best_nn)
    self.Beta = self._project(X, T, solve=True)[2]
#    print "%d of %d neurons selected with a LOO validation" % (len(best_nn), nn)
#    if len(best_nn) > nn*0.9:
#        print "Hint: try re-training with more hidden neurons"


































