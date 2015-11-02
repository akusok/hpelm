# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from tables import open_file
import cPickle
import os
import platform
from solver.solver import Solver


class SLFN(object):
    """Single-hidden Layer Feed-forward Network.
    """

    def __init__(self, inputs, targets, batch=1000, accelerator=None):
        """Initializes a SLFN with an empty hidden layer.

        :param inputs: number of features in input samples (input dimensionality)
        :param outputs: number of simultaneous predicted outputs
        :param batch: batch size, ELM always runs in batch mode
        """
        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(targets, (int, long)), "Number of outputs must be integer"
        assert batch > 0, "Batch should be positive"

        self.inputs = inputs
        self.targets = targets
        # cannot use a dictionary for neurons, because its iteration order is not defined
        self.neurons = []  # list of all neurons in normal Numpy form
        self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")
        self.batch = int(batch)

        # init solver to solve ELM
        if accelerator is None:  # double precision Numpy solver
            self.solver = Solver(targets)
#        if accelerator == "GPU":
#            self.accelerator = "GPU"
#            self.magma_solver = __import__('acc.gpu_solver', globals(), locals(), ['gpu_solve'], -1)
#            print "using GPU"

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
                # assert isinstance(X, np.ndarray) and
                assert X.dtype.kind not in "OSU", "X must be a numerical numpy array"
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)
            assert len(X.shape) == 2, "X must have 2 dimensions"
            msg = "X has wrong dimensionality: expected %d, found %d" % (self.inputs, X.shape[1])
            assert X.shape[1] == self.inputs, msg

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
                # assert isinstance(T, np.ndarray) and
                assert T.dtype.kind not in "OSU", "T must be a numerical numpy array"
                if len(T.shape) == 1:
                    T = T.reshape(-1, 1)
            assert len(T.shape) == 2, "T must have 2 dimensions"
            msg = "T has wrong dimensionality: expected %d, found %d" % (self.targets, T.shape[1])
            assert T.shape[1] == self.targets, msg

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
        assert (func in self.flist or isinstance(func, np.ufunc)),\
            "'%s' neurons not suppored: use a standard neuron function or a custom <numpy.ufunc>" % func
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
        msg = "W must be size [inputs, neurons] (expected [%d,%d])" % (self.inputs, number)
        assert W.shape == (self.inputs, number), msg
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
        self.solver.set_neurons(self.neurons)  # send new neurons to solver


    def project(self, X):
        """Call solver's function.
        """
        X, _ = self._checkdata(X, None)
        H = self.solver.project(X)
        return H


    def predict(self, X):
        """Predict targets for the given inputs X.
        """
        X, _ = self._checkdata(X, None)
        Y = self.solver.predict(X)
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

    ##################
    # helper methods #

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
        self.solver.set_neurons(self.neurons)  # send new neurons to solver

    def _ranking(self, nn):
        """Returns a random ranking of hidden neurons.
        """
        rank = np.arange(nn)
        np.random.shuffle(rank)
        return rank, nn

    def _error(self, Y, T, R=None):
        """Returns regression/classification/multiclass error, also for PRESS.
        """
        raise NotImplementedError("SLFN does not know the use case to compute an error")

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
             "norm": self.solver.norm,
             "Beta": self.solver.get_B(),
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
        self.solver.set_neurons(m["neurons"])
        try:
            self.solver.norm = m["norm"]
        except:
            pass
        self.solver.B = m["Beta"]
        self.classification = m["Classification"]
        self.weights_wc = m["Weights_WC"]


























