# -*- coding: utf-8 -*-
"""HP-ELM iterative solver, just interface. 

Created on Sun Sep  6 11:18:55 2015
@author: akusok
"""

import numpy as np



class Solver(object):

    def __init__(self, neurons, c, norm=None, precision=np.float64):
        """Initialize matrices and functions.
        
        Set neurons to the required precision. Neurons is a list 
            of [('function_type', 'number of neurons', W, Bias), ...]
        Create transformation functions.
        Initialize HH, HT and B matrices, add 'norm' to diagonal of HH.
        """
        if norm is None:
            norm = 50*np.finfo(precision).eps  # 50 times machine epsilon
        self.norm = norm
        self.c = c
        self.L = sum([n[1] for n in neurons])  # total number of neurons
        self.neurons = neurons
        self.to_precision = lambda a: a.astype(precision)

        # apply precision to W and bias in neurons
        self.neurons = []
        for n in neurons:
            W = n[2].astype(precision)
            b = n[3].astype(precision)
            self.neurons.append((n[0], n[1], W, b))

        # transformation functions in HPELM, accessible by name
        self.func = {}
        self.func['lin'] = lambda a: a
        self.func['sigm'] = lambda a: 1/(1 + np.exp(a))
        self.func['tanh'] = lambda a: np.tanh(a)

        # persisitent storage, triangular symmetric matrix
        self.HH = np.zeros((self.L, self.L), dtype=precision)
        self.HT = np.zeros((self.L, self.c), dtype=precision)
        np.fill_diagonal(self.HH, self.norm)
        self.B = None

    def _project(self, X):
        """Projects X to H.
        """
        X = self.to_precision(X)
        return np.hstack([self.func[ftype](np.dot(X, W) + B) for ftype, _, W, B in self.neurons])

    def add_batch(self, X, T):
        """Add a batch of data to an iterative solution.
        """
        H = self._project(X)
        T = self.to_precision(T)
        self.HH += np.dot(H.T, H)
        self.HT += np.dot(H.T, T)

    def solve(self):
        """Compute output weights B, with fix for unstable solution.
        """
        HH_pinv = np.linalg.pinv(self.HH)
        self.B = np.dot(HH_pinv, self.HT)
        return self.B

    def get_corr(self):
        """Return current correlation matrices.
        """
        return self.HH, self.HT

    def predict(self, X):
        """Predict a batch of data.
        """
        assert self.B is not None, "Solve the task before predicting"
        H = self._project(X)
        Y = np.dot(H, self.B)
        return Y




if __name__ == "__main__":
    N = 90
    d = 90
    k = 1
    L = 90/3
    c = 2

    neurons = [('lin', L, np.random.rand(d, L), np.random.rand(L)),
               ('lin', L, np.random.rand(d, L), np.random.rand(L)),
               ('lin', L, np.random.rand(d, L), np.random.rand(L))]

    sl = Solver(neurons, c, precision=np.float64)
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

















#
#if __name__ == "__main__":
#    N = 10000
#    d = 25
#    k = 10
#    L = 60/3
#    c = 2
#
#    neurons = [('lin', L, np.random.rand(d, L), np.random.rand(L)),
#               ('tanh', L, np.random.rand(d, L), np.random.rand(L)),
#               ('sigm', L, np.random.rand(d, L), np.random.rand(L))]
#
#    sl = Solver(neurons, c, precision=np.float64)
#    for _ in xrange(k):
#        X = np.random.rand(N, d)
#        T = np.random.rand(N, c)
#        sl.add_batch(X, T)
#        sl.solve()
#        Y = sl.predict(X)
#        print np.mean((T - Y)**2)
#
#    print "Done!"






























