# -*- coding: utf-8 -*-
"""HP-ELM iterative solver, just interface.

Created on Sun Sep  6 11:18:55 2015
@author: akusok
"""

import numpy as np
from scipy.spatial.distance import cdist


class Solver(object):

    def __init__(self, c, norm=None, precision=np.float64):
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
        self.precision = precision
        self.to_precision = lambda a: a.astype(precision)
        self.neurons = None
        self.B = None

        # transformation functions in HPELM, accessible by name
        self.func = {}
        self.func["lin"] = lambda X, W, B: np.dot(X, W) + B
        self.func["sigm"] = lambda X, W, B: 1 / (1 + np.exp(np.dot(X, W) + B))
        self.func["tanh"] = lambda X, W, B: np.tanh(np.dot(X, W) + B)
        self.func["rbf_l1"] = lambda X, W, B: np.exp(-cdist(X, W.T, "cityblock")**2 / B)
        self.func["rbf_l2"] = lambda X, W, B: np.exp(-cdist(X, W.T, "euclidean")**2 / B)
        self.func["rbf_linf"] = lambda X, W, B: np.exp(-cdist(X, W.T, "chebyshev")**2 / B)

    def _project(self, X):
        """Projects X to H, build-in function.
        """
        X = self.to_precision(X)
        return np.hstack([self.func[ftype](X, W, B) for ftype, _, W, B in self.neurons])

    def project(self, X):
        """Projects X to H and returns a Numpy array.
        """
        assert self.neurons is not None, "ELM has no neurons"
        return self._project(X)

    def add_batch(self, X, T, w=1.0):
        """Add a weighted batch of data to an iterative solution.

        :param w: can be one number or a vector of weights.
        """
        assert self.neurons is not None, "ELM has no neurons"
        w = np.array(w, dtype=self.precision)
        H = self._project(X)
        T = self.to_precision(T)
        self.HH += np.dot(H.T, H) * w
        self.HT += np.dot(H.T, T) * w

    def get_batch(self, X, T, w=1.0):
        """Compute and return a weighted batch of data.

        :param w: can be one number or a vector of weights.
        """
        assert self.neurons is not None, "ELM has no neurons"
        H = self._project(X)
        T = self.to_precision(T)
        w = np.array(w, dtype=self.precision)
        HH = np.dot(H.T, H) * w + np.eye(self.L) * self.norm
        HT = np.dot(H.T, T) * w
        return HH, HT

    def solve(self):
        """Compute output weights B, with fix for unstable solution.
        """
        HH_pinv = np.linalg.pinv(self.HH)
        self.B = np.dot(HH_pinv, self.HT)

    def solve_corr(self, HH, HT):
        """Compute output weights B for given HH and HT.
        """
        HH_pinv = np.linalg.pinv(HH)
        B = np.dot(HH_pinv, HT)
        return B

    def predict(self, X):
        """Predict a batch of data.
        """
        assert self.B is not None, "Solve the task before predicting"
        H = self._project(X)
        Y = np.dot(H, self.B)
        return Y

    def reset(self):
        """Reset current HH, HT and Beta but keep neurons.
        """
        self.HH = np.zeros((self.L, self.L), dtype=self.precision)
        self.HT = np.zeros((self.L, self.c), dtype=self.precision)
        np.fill_diagonal(self.HH, self.norm)
        self.B = None

    ###########################################################
    # setters and getters

    def set_neurons(self, neurons):
        """Set new neurons in solver.
        """
        self.L = sum([n[1] for n in neurons])  # total number of neurons
        # apply precision to W and bias in neurons
        self.neurons = []
        for n in neurons:
            W = self.to_precision(n[2])
            b = self.to_precision(n[3])
            self.neurons.append((n[0], n[1], W, b))
        # init neurons-dependent storage
        self.reset()

    def get_B(self):
        """Return B as a numpy array.
        """
        return self.B

    def get_corr(self):
        """Return current correlation matrices.
        """
        return self.HH, self.HT

    def set_corr(self, HH, HT):
        """Set pre-computed correlation matrices.
        """
        assert self.neurons is not None, "Add or load neurons before using ELM"
        assert HH.shape[0] == HH.shape[1], "HH must be a square matrix"
        msg = "Wrong HH dimension: (%d, %d) expected, %s found" % (self.L, self.L, HH.shape)
        assert HH.shape[0] == self.L, msg
        assert HH.shape[0] == HT.shape[0], "HH and HT must have the same number of rows (%d)" % self.L
        assert HT.shape[1] == self.c, "Number of columns in HT must equal number of targets (%d)" % self.c
        self.HH = self.to_precision(HH)
        self.HT = self.to_precision(HT)


if __name__ == "__main__":
    N = 1000
    d = 15
    k = 10
    L = 100/3
    c = 2

    neurons = [('lin', L, np.random.rand(d, L), np.random.rand(L)),
               ('sigm', L, np.random.rand(d, L), np.random.rand(L)),
               ('tanh', L, np.random.rand(d, L), np.random.rand(L))]

    sl = Solver(c, precision=np.float64)
    sl.set_neurons(neurons)
    for _ in xrange(k):
        X = np.random.rand(N, d)
        T = np.random.rand(N, c)
        sl.add_batch(X, T)

    sl.solve()
    Y = sl.predict(X)
    print (T - Y).shape
    print np.mean((T - Y)**2)
    print "Done!"






    """def project(self, X):
        # assemble hidden layer output with all kinds of neurons
        assert len(self.neurons) > 0, "Model must have hidden neurons"

        X, _ = self._checkdata(X, None)
        H = []
        cdkinds = {"rbf_l2": "euclidean", "rbf_l1": "cityblock", "rbf_linf": "chebyshev"}
        for func, _, W, B in self.neurons:
            # projection
            if "rbf" in func:
                self._affinityfix()
                N = X.shape[0]
                k = cpu_count()
                jobs = [(X[idx], W.T, cdkinds[func], idx) for idx in np.array_split(np.arange(N), k*10)]  #### ERROR HERE!!!
                p = Pool(k)
                H0 = np.empty((N, W.shape[1]))
                for h0, idx in p.imap(cd, jobs):
                    H0[idx] = h0
                p.close()
                H0 = - H0 / B
            else:
                H0 = X.dot(W) + B

            # transformation
            if func == "lin":
                pass
            elif "rbf" in func:
                ne.evaluate('exp(H0)', out=H0)
            elif func == "sigm":
                ne.evaluate("1/(1+exp(-H0))", out=H0)
            elif func == "tanh":
                ne.evaluate('tanh(H0)', out=H0)
            else:
                H0 = func(H0)  # custom <numpy.ufunc>
            H.append(H0)

        if len(H) == 1:
            H = H[0]
        else:
            H = np.hstack(H)
#        print (H > 0.01).sum(0)
        return H"""





















