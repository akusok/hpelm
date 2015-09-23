# -*- coding: utf-8 -*-
"""HP-ELM iterative solver in python (numpy etc., CPU only).

Created on Sun Sep  6 11:19:09 2015

@author: akusok
"""

from solver import Solver
import numpy as np
from scipy.linalg import blas, lapack
import numexpr as ne


class SolverPython(Solver):

    def __init__(self, neurons, c, norm=None, precision=np.float64):
        """Do all differences in initialization.
        """
        super(SolverPython, self).__init__(neurons, c, norm, precision)

        # get correct BLAS/LAPACK functions for precision
        if precision is np.float32:
            self.syrk = blas.ssyrk
            self.posv = lapack.sposv
        elif precision is np.float64:
            self.syrk = blas.dsyrk
            self.posv = lapack.dposv
        else:
            raise NotImplementedError("Only single and double precision supported")

        # transformation functions in HPELM, accessible by name
        self.func['sigm'] = lambda a: ne.evaluate("1/(1+exp(-a))").astype(precision)

        # persisitent storage, triangular symmetric matrix
        self.HH = np.zeros((self.L, self.L), dtype=precision, order='F')
        self.HT = np.zeros((self.L, self.c), dtype=precision, order='F')
        np.fill_diagonal(self.HH, self.norm)

    def add_batch(self, X, T):
        """Add a batch of data to iterative solution
        """
        H = self._project(X)
        T = self.to_precision(T)
        self.syrk(1, H.T, 1, self.HH, trans=0, overwrite_c=1)  # self.HH += np.dot(H.T, H)
        self.HT += np.dot(H.T, T)

    def solve(self):
        """Compute output weights B, with fix for unstable solution.
        """
        _, self.B, info = self.posv(self.HH, self.HT)
        if info > 0:
            print "Covariance matrix is not full rank; solving with SVD (slow)"
            print "This happened because you have duplicated or too many neurons"
            HH = self.HH + np.triu(self.HH, k=1).T
            self.B = np.linalg.lstsq(HH, self.HT)[0]
        return self.B



if __name__ == "__main__":
    N = 99
    d = 3
    k = 1
    L = 99
    c = 2

    neurons = [('tanh', L, np.random.randn(d, L)/10, np.random.rand(L))]#,
               #('lin', L, np.random.randn(d, L), np.random.rand(L)),
               #('lin', L, np.random.randn(d, L), np.random.rand(L))]

    sl = SolverPython(neurons, c, precision=np.float64)
    for _ in xrange(k):
        X = np.hstack((np.random.rand(N, d-1), np.ones((N,1))))
        T = np.random.rand(N, c)
        sl.add_batch(X, T)

    B = sl.solve()         
    Y = sl.predict(X)
    print (T - Y).shape
    print np.mean((T - Y)**2)
    
    print "########################"
    H = sl._project(X)
    HH = H.T.dot(H)
    HT = H.T.dot(T)
    A = np.linalg.solve(HH, HT)
    Y = H.dot(A)
    print np.mean((T - Y)**2)
    
    
#    print T-Y2

#    print B1 - A
    print "Done!"




































