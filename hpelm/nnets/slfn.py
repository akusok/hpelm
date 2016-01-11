# -*- coding: utf-8 -*-
"""HP-ELM iterative solver, just interface.

Created on Sun Sep  6 11:18:55 2015
@author: akusok
"""

import numpy as np
from scipy.spatial.distance import cdist
import platform
import os


class SLFN(object):
    """Single Layer Feed-forward Network (SLFN), the neural network that ELM trains.

    This implementation is not the fastest but very simple, and it defines interface.

    Args:
        c (int): number of outputs, or classes for classification
        norm (double): output weights normalization parameter (Tikhonov normalizaion, or
            ridge regression), large values provides smaller (= better) weights but worse model accuracy
        precision (Numpy.float32/64): solver precision, float32 is faster but may be worse, most GPU
            work fast only in float32.

    Attributes:
        neurons (list): a list of different types of neurons, initially empty. One neuron type is a tuple
            ('number of neurons', 'function_type', W, Bias), `neurons` is a list of [neuron_type_1, neuron_type_2, ... ].
        func (dict): a dictionary of transformation function type, key is a neuron type (= function name)
            and value is the function itself. A single function takes input parameters X, W, B, and outputs
            corresponding H for its neuron type.
        HH, HT (matrix): intermediate covariance matrices used in ELM solution. Can be computed and stored in GPU
            memory for accelerated SLFN. They are not needed once ELM is solved and they can take a lot of memory
            with large numbers of neurons, so one can delete them with `reset()` method. They are omitted when
            an ELM model is saved.
        B (matrix): output solution matrix of SLFN. A trained ELM needs only `neurons` and `B` to predict outputs
            for new input data.
    """

    def __init__(self, c, norm=None, precision=np.float64):
        """Initialize class variables and transformation functions.
        """
        self.c = c  # number of outputs, also number of classes (thus 'c')
        self.precision = precision
        if norm is None:
            norm = 50*np.finfo(precision).eps  # 50 times machine epsilon
        self.norm = precision(norm)  # cast norm in required precision
        # cannot use a dictionary for neurons, because its iteration order is not defined
        self.neurons = []  # list of all neurons in normal Numpy form
        self.L = None  # number of neurons
        self.B = None
        self.HH = None
        self.HT = None

        # transformation functions in HPELM, accessible by name
        self.func = {}
        self.func["lin"] = lambda X, W, B: np.dot(X, W) + B
        self.func["sigm"] = lambda X, W, B: 1 / (1 + np.exp(np.dot(X, W) + B))
        self.func["tanh"] = lambda X, W, B: np.tanh(np.dot(X, W) + B)
        self.func["rbf_l1"] = lambda X, W, B: np.exp(-cdist(X, W.T, "cityblock")**2 / B)
        self.func["rbf_l2"] = lambda X, W, B: np.exp(-cdist(X, W.T, "euclidean")**2 / B)
        self.func["rbf_linf"] = lambda X, W, B: np.exp(-cdist(X, W.T, "chebyshev")**2 / B)

    def add_neurons(self, number, func, W, B):
        """Add prepared neurons to the SLFN, merge with existing ones.

        Adds a number of specific neurons to SLFN network. Weights and biases
        are generated automatically if not provided, but they assume that input
        data is normalized (input data features have zero mean and unit variance).

        If neurons of such type already exist, they are merged together.

        Parameters
        ----------
            number : int
                A number of new neurons to add
            func : {'lin', 'sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf'}
                Transformation function of hidden layer. Linear function leads
                to a linear model.
            W : array_like, optional
                A 2-D matrix of neuron weights, size (`inputs`, `number`)
            B : array_like, optional
                A 1-D vector of neuron biases, size (`number`, )
        """

        ntypes = [nr[1] for nr in self.neurons]  # existing types of neurons
        if func in ntypes:
            # add to an existing neuron type
            i = ntypes.index(func)
            nn0, _, W0, B0 = self.neurons[i]
            number = nn0 + number
            W = np.hstack((W0, W))
            B = np.hstack((B0, B))
            self.neurons[i] = (number, func, W, B)
        else:
            # create a new neuron type
            self.neurons.append((number, func, W, B))
        self.reset()
        self.B = None

    def reset(self):
        """ Resets intermediate training results, frees memory that they use.

        Keeps solution of ELM, so a trained ELM remains operational.
        Can be called to free memory after an ELM is trained.
        """
        self.L = sum([n[0] for n in self.neurons])  # get number of neurons
        self.HH = None
        self.HT = None

    def project(self, X):
        """Projects X to H, build-in function.
        """
        assert self.neurons is not None, "ELM has no neurons"
        X = X.astype(self.precision)
        return np.hstack([self.func[ftype](X, W, B) for _, ftype, W, B in self.neurons])

    def predict(self, X):
        """Predict a batch of data.
        """
        assert self.B is not None, "Solve the task before predicting"
        H = self.project(X)
        Y = np.dot(H, self.B)
        return Y

    def add_batch(self, X, T, wc = None):
        """Add a weighted batch of data to an iterative solution.

        :param wc: vector of weights for data samples, same length as X or T
        """
        H = self.project(X)
        T = T.astype(self.precision)
        if wc is not None:  # apply weights if given
            w = np.array(wc**0.5, dtype=self.precision)[:, None]  # re-shape to column matrix
            H *= w
            T *= w

        if self.HH is None:  # initialize space for self.HH, self.HT
            self.HH = np.zeros((self.L, self.L), dtype=self.precision)
            self.HT = np.zeros((self.L, self.c), dtype=self.precision)
            np.fill_diagonal(self.HH, self.norm)

        self.HH += np.dot(H.T, H)
        self.HT += np.dot(H.T, T)

    def get_batch(self, X, T, wc = None):
        """Compute and return a weighted batch of data.

        :param wc: vector of weights for data samples, same length as X or T
        """
        H = self.project(X)
        T = T.astype(self.precision)
        if wc is not None:  # apply weights if given
            w = np.array(wc**0.5, dtype=self.precision)[:, None]  # re-shape to column matrix
            H *= w
            T *= w
        HH = np.dot(H.T, H) + np.eye(H.shape[1]) * self.norm
        HT = np.dot(H.T, T)
        return HH, HT

    def solve(self):
        """Redirects to solve_corr, to avoid duplication of code.
        """
        self.B = self.solve_corr(self.HH, self.HT)

    def solve_corr(self, HH, HT):
        """Compute output weights B for given HH and HT.

        Simple but inefficient version, see a better one in solver_python.
        """
        HH_pinv = np.linalg.pinv(HH)
        B = np.dot(HH_pinv, HT)
        return B

    def _prune(self, idx):
        """Leave only neurons with the given indexes.
        """
        idx = list(idx)
        neurons = []
        for nold in self.neurons:
            k = nold[0]  # number of neurons
            ix1 = [i for i in idx if i < k]  # index for current neuron type
            idx = [i-k for i in idx if i >= k]
            func = nold[1]
            number = len(ix1)
            W = nold[2][:, ix1]
            bias = nold[3][ix1]
            neurons.append((number, func, W, bias))
        self.neurons = neurons
        # reset invalid parameters
        self.reset()
        self.B = None




    ###########################################################
    # setters and getters

    def get_B(self):
        """Return B as a numpy array.
        """
        return self.B

    def set_B(self, B):
        """Set B as a numpy array.

        :param B: output layer weights matrix.
        """
        assert B.shape[0] == self.L, "Incorrect first dimension: %d expected, %d found" % (self.L, B.shape[0])
        assert B.shape[1] == self.c, "Incorrect output dimension: %d expected, %d found" % (self.c, B.shape[1])
        self.B = B.astype(self.precision)

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



    def fix_affinity(self):
        """Numpy processor core affinity fix.

        Fixes a problem when all Numpy processes are pushed to core 0.
        """
        if "Linux" in platform.system():
            a = np.random.rand(3, 1)
            np.dot(a.T, a)
            pid = os.getpid()
            os.system("taskset -p 0xffffffff %d >/dev/null" % pid)















