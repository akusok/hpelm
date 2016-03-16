# -*- coding: utf-8 -*-
"""Basic SLFN solver that follows paper notations, defines interface for all solvers.


Created on Sun Sep  6 11:18:55 2015
@author: akusok
"""

import os
import platform

import numpy as np
from scipy.spatial.distance import cdist


class SLFN(object):
    """Single Layer Feed-forward Network (SLFN), the neural network that ELM trains.

    This implementation is not the fastest but very simple, and it defines interface.
    Gives correct output, other solvers should provide the same output as this guy.

    Args:
        outputs (int): number of outputs, or classes for classification
        norm (double): output weights normalization parameter (Tikhonov normalizaion, or
            ridge regression), large values provides smaller (= better) weights but worse model accuracy
        precision (Numpy.float32/64): solver precision, float32 is faster but may be worse, most GPU
            work fast only in float32.

    Attributes:
        neurons (list): a list of different types of neurons, initially empty. One neuron type is a tuple
            ('number of neurons', 'function_type', W, Bias), `neurons` is a list of [neuron_type_1, neuron_type_2, ...].
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

    def __init__(self, inputs, outputs, norm=None, precision=np.float64):
        """Initialize class variables and transformation functions.
        """
        self.inputs = inputs  # not used here, but logically are a part of SLFN
        self.outputs = outputs  # number of outputs, also number of classes (thus 'c')
        self.precision = precision
        if norm is None:
            norm = 50*np.finfo(precision).eps  # 50 times machine epsilon
        self.norm = norm
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
        must be provided for that function.

        If neurons of such type already exist, they are merged together.

        Args:
            number (int): the number of new neurons to add
            func (str): transformation function of hidden layer. Linear function creates a linear model.
            W (matrix): a 2-D matrix of neuron weights, size (`inputs` * `number`)
            B (vector): a 1-D vector of neuron biases, size (`number` * 1)
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
        """ Resets intermediate training results, releases memory that they use.

        Keeps solution of ELM, so a trained ELM remains operational.
        Can be called to free memory after an ELM is trained.
        """
        self.L = sum([n[0] for n in self.neurons])  # get number of neurons
        self.HH = None
        self.HT = None

    def _project(self, X):
        """Projects X to H, an auxiliary function that implements a particular projection.

        For actual projection, use `ELM.project()` instead.

        Args:
            X (matrix): an input data matrix, size (N * `inputs`)

        Returns:
            H (matrix): an SLFN hidden layer representation, size (N * `L`) where 'L' is number of neurons
        """
        assert self.neurons is not None, "ELM has no neurons"
        return np.hstack([self.func[ftype](X, W, B) for _, ftype, W, B in self.neurons])

    def _predict(self, X):
        """Predict a batch of data. Auxiliary function that implements a particular prediction.

        For prediction, use `ELM.predict()` instead.

        Args:
            X (matrix): input data size (N * `inputs`)

        Returns:
            Y (matrix): predicted outputs size (N * `outputs`), always in float/double format.
        """
        assert self.B is not None, "Solve the task before predicting"
        H = self._project(X)
        Y = np.dot(H, self.B)
        return Y

    def add_batch(self, X, T, wc=None):
        """Add a batch of training data to an iterative solution, weighted if neeed.

        The batch is processed as a whole, the training data is splitted in `ELM.add_data()` method.
        With parameters HH_out, HT_out, the output will be put into these matrices instead of model.

        Args:
            X (matrix): input data matrix size (N * `inputs`)
            T (matrix): output data matrix size (N * `outputs`)
            wc (vector): vector of weights for data samples, one weight per sample, size (N * 1)
            HH_out, HT_out (matrix, optional): output matrices to add batch result into, always given together
        """
        H = self._project(X)
        T = T.astype(self.precision)
        if wc is not None:  # apply weights if given
            w = np.array(wc**0.5, dtype=self.precision)[:, None]  # re-shape to column matrix
            H *= w
            T *= w

        if self.HH is None:  # initialize space for self.HH, self.HT
            self.HH = np.zeros((self.L, self.L), dtype=self.precision)
            self.HT = np.zeros((self.L, self.outputs), dtype=self.precision)
            np.fill_diagonal(self.HH, self.norm)

        self.HH += np.dot(H.T, H)
        self.HT += np.dot(H.T, T)

    def solve(self):
        """Redirects to solve_corr, to avoid duplication of code.
        """
        self.B = self.solve_corr(self.HH, self.HT)

    def solve_corr(self, HH, HT):
        """Compute output weights B for given HH and HT.

        Simple but inefficient version, see a better one in solver_python.

        Args:
            HH (matrix): covariance matrix of hidden layer represenation H, size (`L` * `L`)
            HT (matrix): correlation matrix between H and outputs T, size (`L` * `outputs`)
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

    def get_B(self):
        """Return B as a numpy array.
        """
        return self.B

    def set_B(self, B):
        """Set B as a numpy array.

        Args:
            B (matrix): output layer weights matrix, size (`L` * `outputs`)
        """
        assert isinstance(B, np.ndarray), "B should be a Numpy ndarray"
        assert len(B.shape) > 0, "Cannot set empty B"
        assert B.shape[0] == self.L, "Incorrect first dimension: %d expected, %d found" % (self.L, B.shape[0])
        assert B.shape[1] == self.outputs, "Incorrect output dimension: %d expected, %d found" % (self.outputs, B.shape[1])
        self.B = B.astype(self.precision)

    def get_corr(self):
        """Return current correlation matrices.
        """
        return self.HH, self.HT

    def set_corr(self, HH, HT):
        """Set pre-computed correlation matrices.

        Args:
            HH (matrix): covariance matrix of hidden layer represenation H, size (`L` * `L`)
            HT (matrix): correlation matrix between H and outputs T, size (`L` * `outputs`)
        """
        assert self.neurons is not None, "Add or load neurons before using ELM"
        assert HH.shape[0] == HH.shape[1], "HH must be a square matrix"
        msg = "Wrong HH dimension: (%d, %d) expected, %s found" % (self.L, self.L, HH.shape)
        assert HH.shape[0] == self.L, msg
        assert HH.shape[0] == HT.shape[0], "HH and HT must have the same number of rows (%d)" % self.L
        assert HT.shape[1] == self.outputs, "Number of columns in HT must equal number of outputs (%d)" % self.outputs
        self.HH = HH.astype(self.precision)
        self.HT = HT.astype(self.precision)

    def get_neurons(self):
        """Return current neurons.

        Returns:
            neurons (list of tuples (number/int, func/string, W/matrix, B/vector)): current neurons in the model
        """
        return self.neurons

    def fix_affinity(self):
        """Numpy processor core affinity fix.

        Fixes a problem if all Numpy processes are pushed to CPU core 0.
        """
        if "Linux" in platform.system():
            a = np.random.rand(3, 1)
            np.dot(a.T, a)
            pid = os.getpid()
            os.system("taskset -p 0xffffffff %d >/dev/null" % pid)
