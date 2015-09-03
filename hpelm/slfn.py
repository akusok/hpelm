# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import numexpr as ne
from tables import open_file
from scipy.spatial.distance import cdist
from scipy.linalg import solve as cpu_solve
from multiprocessing import Pool, cpu_count
import cPickle
import os, platform


def cd(a):
    x, w, kind, idx = a
    return cdist(x, w, kind)**2, idx


class SLFN(object):
    """Single-hidden Layer Feed-forward Network.
    """

    def __init__(self, inputs, targets, batch=100, accelerator=""):
        """Initializes a SLFN with an empty hidden layer.

        :param inputs: number of features in input samples (input dimensionality)
        :param outputs: number of simultaneous predicted outputs
        """
        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(targets, (int, long)), "Number of outputs must be integer"
        assert batch > 0, "Batch should be positive"

        self.inputs = inputs
        self.targets = targets
        # cannot use a dictionary for neurons, because its iteration order is not defined
        self.neurons = []  # list of all neurons with their types (= transformantion functions)
        self.Beta = None
        self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")
        self.alpha = 1E-9  # normalization for H'H solution
        self.batch = int(batch)  # smallest batch for batch processing
        self.accelerator = None  # None, "GPU", "PHI"
        if accelerator == "GPU":
            self.accelerator = "GPU"
            self.magma_solver = __import__('acc.gpu_solver', globals(), locals(), ['gpu_solve'], -1)
            print "using GPU"
        # init other stuff
        self.opened_hdf5 = []
        self.classification = None  # c / wc / mc
        self.weights_wc = None
        self.tprint = 5

    def __del__(self):
        """Close HDF5 files opened during HPELM usage.
        """
        if len(self.opened_hdf5) > 0:
            for h5 in self.opened_hdf5:
                h5.close()

    def _checkdata(self, X, T):
        """Checks data variables and fixes matrix dimensionality issues.
        """
        if X is not None:
            if isinstance(X, basestring):  # open HDF5 file
                try:
                    h5 = open_file(X, "r")
                    self.opened_hdf5.append(h5)
                    for node in h5.walk_nodes():
                        pass  # find a node with whatever name
                    X = node
                except:
                    raise IOError("Cannot read HDF5 file at %s" % X)
            else:
                # assert isinstance(X, np.ndarray) and X.dtype.kind not in "OSU", "X must be a numerical numpy array"
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)
            assert len(X.shape) == 2, "X must have 2 dimensions"
            assert X.shape[1] == self.inputs, "X has wrong dimensionality: expected %d, found %d" % (self.inputs, X.shape[1])

        if T is not None:
            if isinstance(T, basestring):  # open HDF5 file
                try:
                    h5 = open_file(T, "r")
                    self.opened_hdf5.append(h5)
                    for node in h5.walk_nodes():
                        pass  # find a node with whatever name
                    T = node
                except:
                    raise IOError("Cannot read HDF5 file at %s" % T)
            else:
                # assert isinstance(T, np.ndarray) and T.dtype.kind not in "OSU", "T must be a numerical numpy array"
                if len(T.shape) == 1:
                    T = T.reshape(-1, 1)
            assert len(T.shape) == 2, "T must have 2 dimensions"
            assert T.shape[1] == self.targets, "T has wrong dimensionality: expected %d, found %d" % (self.targets, T.shape[1])

        if (X is not None) and (T is not None):
            assert X.shape[0] == T.shape[0], "X and T cannot have different number of samples"

        return X, T

    def add_neurons(self, number, func, W=None, B=None):
        """Add neurons of a specific type to the SLFN.

        If neurons of such type exist, merges them together.
        :param number: - number of neurons to add
        :param func: - transformation function of those neurons,
                       "lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf"
                       or a custom function of type <numpy.ufunc>
        :param W: - projection matrix or ("rbf_xx") a list of centroids
        :param B: - bias vector or ("rbf_xx") a vector of sigmas
        """
        assert isinstance(number, int), "Number of neurons must be integer"
        assert func in self.flist or isinstance(func, np.ufunc), "Use standard neuron function or a custom <numpy.ufunc>"
        assert isinstance(W, (np.ndarray, type(None))), "Projection matrix (W) must be a Numpy ndarray"
        assert isinstance(B, (np.ndarray, type(None))), "Bias vector (B) must be a Numpy ndarray"

        # initialize skipped arguments
        if W is None:
            if func == "lin":  # copying input features for linear neurons
                number = min(number, self.inputs)  # cannot have more linear neurons than features
                W = np.eye(self.inputs, number)
            else:
                W = np.random.randn(self.inputs, number)
                if func not in ("rbf_l1", "rbf_l2", "rbf_linf"):
                    W = W * (3 / self.inputs ** 0.5)  # high dimensionality fix
        if B is None:
            B = np.random.randn(number)
            if func in ("rbf_l2", "rbf_l1", "rbf_linf"):
                B = np.abs(B)
                B = B * self.inputs
            if func == "lin":
                B = np.zeros((number,))
        assert W.shape == (self.inputs, number), "W must be size [inputs, neurons] (expected [%d,%d])" % (self.inputs, number)
        assert B.shape == (number,), "B must be size [neurons] (expected [%d])" % number

        ntypes = [nr[0] for nr in self.neurons]  # existing types of neurons
        if func in ntypes:
            # add to an existing neuron type
            i = ntypes.index(func)
            _, nn0, W0, B0 = self.neurons[i]
            number = nn0 + number
            W = np.hstack((W0, W))
            B = np.hstack((B0, B))
            self.neurons[i] = (func, number, W, B)
        else:
            # create a new neuron type
            self.neurons.append((func, number, W, B))
        self.Beta = None  # need to re-train network after adding neurons


    def project(self, X):
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
#            if func == "rbf_l2":
#                H0 = - cdist(X, W.T, "euclidean")**2 / B
#            elif func == "rbf_l1":
#                H0 = - cdist(X, W.T, "cityblock")**2 / B
#            elif func == "rbf_linf":
#                H0 = - cdist(X, W.T, "chebyshev")**2 / B
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
        return H

    def predict(self, X):
        """Predict targets for the given inputs X.

        :param X: - model inputs
        """
        assert self.Beta is not None, "Train ELM before predicting"
        H = self.project(X)
        Y = H.dot(self.Beta)
        return Y

    def error(self, Y, T):
        """Calculate error of model predictions.
        """
        _, Y = self._checkdata(None, Y)
        _, T = self._checkdata(None, T)
        return self._error(Y, T)

    def confusion(self, Y1, T1):
        """Compute confusion matrix for the given classification, iteratively.
        """
        _, Y = self._checkdata(None, Y1)
        _, T = self._checkdata(None, T1)
        nn = np.sum([n1[1] for n1 in self.neurons])
        N = T.shape[0]
        batch = max(self.batch, nn)
        nb = N / batch  # number of batches
        if batch > N * nb:
            nb += 1

        C = self.targets
        conf = np.zeros((C, C))

        if self.classification in ("c", "wc"):
            for b in xrange(nb):
                start = b*batch
                stop = min((b+1)*batch, N)
                Tb = np.array(T[start:stop]).argmax(1)
                Yb = np.array(Y[start:stop]).argmax(1)
                for c1 in xrange(C):
                    for c1h in xrange(C):
                        conf[c1, c1h] += np.sum((Tb == c1) * (Yb == c1h))
        elif self.classification == "mc":
            for b in xrange(nb):
                start = b*batch
                stop = min((b+1)*batch, N)
                Tb = np.array(T[start:stop]) > 0.5
                Yb = np.array(Y[start:stop]) > 0.5
                for c1 in xrange(C):
                    for c1h in xrange(C):
                        conf[c1, c1h] += np.sum(Tb[:, c1] * Yb[:, c1h])
        else:  # No confusion matrix
            conf = None
        return conf

    ######################
    ### helper methods ###

    def _prune(self, idx):
        """Leave only neurons with the given indexes.
        """
        idx = list(idx)
        neurons = []
        for nold in self.neurons:
            k = nold[1]  # number of neurons
            ix1 = [i for i in idx if i < k]  # index for current neuron type
            idx = [i-k for i in idx if i >= k]
            func = nold[0]
            number = len(ix1)
            W = nold[2][:, ix1]
            bias = nold[3][ix1]
            neurons.append((func, number, W, bias))
        self.neurons = neurons

    def _ranking(self, nn):
        """Returns a random ranking of hidden neurons.
        """
        rank = np.arange(nn)
        np.random.shuffle(rank)
        return rank, nn

    def _solve_corr(self, HH, HT):
        """Solve a linear system from correlation matrices.
        """
        if self.accelerator == "GPU":
            Beta = self.magma_solver.gpu_solve(HH, HT, self.alpha)
        else:
            Beta = cpu_solve(HH, HT, sym_pos=True)
        return Beta

    def _error(self, Y, T, R=None):
        """Returns regression/classification/multiclass error, also for PRESS.
        """
        raise NotImplementedError("SLFN does not know the use case to compute an error")

    def _train(self, X, T):
        raise NotImplementedError("SLFN does not know the use case to train a network")

    def __str__(self):
        s = "SLFN with %d inputs and %d outputs\n" % (self.inputs, self.targets)
        s += "Hidden layer neurons: "
        for func, n, _, _ in self.neurons:
            s += "%d %s, " % (n, func)
        s = s[:-2]
        return s

    def _affinityfix(self):
        # Numpy processor affinity fix
        if "Linux" in platform.system():
            a = np.random.rand(3, 1)
            np.dot(a.T, a)
            pid = os.getpid()
            os.system("taskset -p 0xffffffff %d >/dev/null" % pid)

    def save(self, fname):
        assert isinstance(fname, basestring), "Model file name must be a string"
        m = {"inputs": self.inputs,
             "outputs": self.targets,
             "neurons": self.neurons,
             "Beta": self.Beta,
             "alpha": self.alpha,
             "Classification": self.classification,
             "Weights_WC": self.weights_wc}
        try:
            cPickle.dump(m, open(fname, "wb"), -1)
        except IOError:
            raise IOError("Cannot create a model file at: %s" % fname)

    def load(self, fname):
        assert isinstance(fname, basestring), "Model file name must be a string"
        try:
            m = cPickle.load(open(fname, "rb"))
        except IOError:
            raise IOError("Model file not found: %s" % fname)
        self.inputs = m["inputs"]
        self.targets = m["outputs"]
        self.neurons = m["neurons"]
        self.Beta = m["Beta"]
        self.alpha = m["alpha"]
        self.classification = m["Classification"]
        self.weights_wc = m["Weights_WC"]


























