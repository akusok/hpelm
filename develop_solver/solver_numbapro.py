# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:40:43 2015

@author: Anton
"""

from solver import Solver
import numpy as np
from scipy.linalg import lapack
from numbapro.cudalib import cublas


class SolverNumbapro(Solver):

    def __init__(self, neurons, c, norm=None, precision=np.float64):
        """#########################################################"""
        assert precision in (np.float32, np.float64), \
            "Only single or double precision (numpy.float32, numpy.float64) are supported"
        super(SolverNumbapro, self).__init__(neurons, c, norm, precision)
        self.HT = np.zeros((self.L, self.c), dtype=precision, order='F')
        self.posv = lapack.sposv if precision is np.float32 else lapack.dposv
        self.blas = cublas.Blas()        

    def add_batch(self, X, T):
        """Add a batch of data to an iterative solution.
        """
        batch = X.shape[0]
        H = np.asfortranarray(self._project(X))
        T = np.asfortranarray(self.to_precision(T))
        self.blas.syrk('L', 'T', self.L, batch, 1, H, 1, self.HH)
        self.blas.gemm('T', 'N', self.L, self.c, batch, 1, H, T, 1, self.HT)

    def solve(self):
        """Compute output weights B, with fix for unstable solution.
        """
        _, self.B, info = self.posv(self.HH, self.HT)
        if info > 0:
            print "Fast solution is numerically unstable, using a slow alternarive"
            HH = self.HH + np.triu(self.HH, k=1).T
            self.B = np.linalg.lstsq(HH, self.HT)[0]
        return self.B


if __name__ == "__main__":
    N = 10000
    d = 25
    k = 10
    L = 60/3
    c = 5

    neurons = [('lin', L, np.random.rand(d, L), np.random.rand(L))]#,
               #('tanh', L, np.random.rand(d, L), np.random.rand(L)),
               #('sigm', L, np.random.rand(d, L), np.random.rand(L))]

    sl = SolverNumbapro(neurons, c, precision=np.float32)
    sl2 = Solver(neurons, c, precision=np.float32)
    for _ in xrange(k):
        X = np.random.rand(N, d)
        T = np.random.rand(N, c)
        sl.add_batch(X, T)
        sl2.add_batch(X, T)
        print sl.HT[:3, :4] - sl2.HT[:3, :4]
        print

    sl.solve()
    Y = sl.predict(X)
    print np.mean((T - Y)**2)

    sl2.solve()
    Y2 = sl2.predict(X)
    print np.mean((T - Y2)**2)


    print "Done!"

























