# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import numexpr as ne
from scipy.spatial.distance import cdist
import cPickle


class SLFN(object):
    """Single-hidden Layer Feed-forward Network.
    """

    inputs = 0
    targets = 0
    # cannot use a dictionary for neurons, because its iteration order is not defined
    neurons = None  # list of all neurons with their types (= transformantion functions)
    Beta = None
    flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")

    def __init__(self, inputs, outputs):
        """Initializes a SLFN with an empty hidden layer.

        :param inputs: number of features in input samples (input dimensionality)
        :param outputs: number of simultaneous predicted outputs
        """
        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(outputs, (int, long)), "Number of outputs must be integer"

        # set default argument values
        self.inputs = inputs
        self.targets = outputs
        self.neurons = []  # create a separate list for each object

    def _checkdata(self, X, T):
        """Checks data variables and fixes matrix dimensionality issues.
        """
        if X is not None:
            assert isinstance(X, np.ndarray) and X.dtype.kind not in "OSU", "X must be a numerical numpy array"
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            assert len(X.shape) == 2, "X must be 2-dimensional matrix"
            assert X.shape[1] == self.inputs, "X has wrong dimensionality: expected %d, found %d" % (self.inputs, X.shape[1])

        if T is not None:
            assert isinstance(T, np.ndarray) and T.dtype.kind not in "OSU", "T must be a numerical numpy array"
            if len(T.shape) == 1:
                T = T.reshape(-1, 1)
            assert len(X.shape) == 2, "T must be 1- or 2-dimensional matrix"
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
            # the following causes errors with very high dimensional inputs
            #if func not in ("rbf_l1", "rbf_l2", "rbf_linf"):
                #B = (np.abs(B) * self.inputs) ** 0.5  # high dimensionality fix
                #B = B * (self.inputs ** 0.5)  # high dimensionality fix
                #pass
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

    def project(self, X):
        # assemble hidden layer output with all kinds of neurons
        assert len(self.neurons) > 0, "Model must have hidden neurons"
        X, _ = self._checkdata(X, None)
        H = []
        for func, _, W, B in self.neurons:
            # projection
            if func == "rbf_l2":
                H0 = cdist(X, W.T, "sqeuclidean") / (-2 * (B ** 2))
            elif func == "rbf_l1":
                H0 = cdist(X, W.T, "cityblock") / (-2 * (B ** 2))
            elif func == "rbf_linf":
                H0 = cdist(X, W.T, "chebyshev") / (-2 * (B ** 2))
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

        H = np.hstack(H)
        return H

    def predict(self, X):
        """Predict targets for the given inputs X.

        :param X: - model inputs
        """
        assert self.Beta is not None, "Train ELM before predicting"
        H = self.project(X)
        Y = H.dot(self.Beta)
        return Y

    ######################
    ### helper methods ###

    def __str__(self):
        s = "SLFN with %d inputs and %d outputs\n" % (self.inputs, self.targets)
        s += "Hidden layer neurons: "
        for func, n, _, _ in self.neurons:
            s += "%d %s, " % (n, func)
        s = s[:-2]
        return s

    def save(self, model):
        assert isinstance(model, basestring), "Model file name must be a string"
        m = {"inputs": self.inputs,
             "outputs": self.targets,
             "neurons": self.neurons,
             "Beta": self.Beta}
        try:
            cPickle.dump(m, open(model, "wb"), -1)
        except IOError:
            raise IOError("Cannot create a model file at: %s" % model)

    def load(self, model):
        assert isinstance(model, basestring), "Model file name must be a string"
        try:
            m = cPickle.load(open(model, "rb"))
        except IOError:
            raise IOError("Model file not found: %s" % model)
        self.inputs = m["inputs"]
        self.targets = m["outputs"]
        self.neurons = m["neurons"]
        self.Beta = m["Beta"]


























