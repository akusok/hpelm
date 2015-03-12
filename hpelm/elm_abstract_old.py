# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import abc
import cPickle


class ELM_abstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def project(self, X):
        # projects X into H
        H = X
        return H

    @abc.abstractmethod
    def train(self, X, T):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    ########################
    # non-abstract methods #

    def __init__(self, inputs, outputs, elm_type=""):
        """Create ELM of desired kind.

        :param regression: type of ELM task, can be 'regression', 'classification' or 'timeseries' regression.
        :param sparse: set to create an ELM with sparse projection matrix.
        """
        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(outputs, (int, long)), "Number of outputs must be integer"
        assert isinstance(elm_type, basestring), "Type of ELM must be a string"

        self.classification = False
        self.multiclass = False
        self.regression = False
        if "classification" in elm_type:
            self.classification = True
        elif "multiclass" in elm_type:
            self.multiclass = True
        else:
            self.regression = True

        # set default argument values
        self.inputs = inputs
        self.targets = outputs
        self.neurons = {}  # list of all neuron types
        self.Beta = None
        self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")

    def _checkdata(self, X, T):
        """Checks data variables, fixes dimensionality issues.
        """
        if X is not None:
            assert isinstance(X, np.ndarray), "X must be a numpy array"
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            assert len(X.shape) == 2, "X must be 2-dimensional matrix"
            assert X.shape[1] == self.inputs, "X has wrong dimensionality: expected %d, found %d" % (self.inputs, X.shape[1])

        if T is not None:
            assert isinstance(T, np.ndarray), "T must be a numpy array"
            if len(T.shape) == 1:
                T = T.reshape(-1, 1)
            assert len(X.shape) == 2, "T must be 1- or 2-dimensional matrix"
            assert T.shape[1] == self.targets, "T has wrong dimensionality: expected %d, found %d" % (self.targets, T.shape[1])

        if (X is not None) and (T is not None):
            assert X.shape[0] == T.shape[0], "X and T cannot have different number of samples"

        return X, T

    def add_neurons(self, number, func, W=None, B=None):
        """Add neurons of a specific type to the ELM model.

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
                W = np.eye(self.inputs, number)
            else:
                W = np.random.randn(self.inputs, number)
                if "rbf" not in func:
                    W = W * (3 / self.inputs ** 0.5)  # high dimensionality fix
        if B is None:
            B = np.random.randn(number)
            if "rbf" in func:
                B = (np.abs(B) * self.inputs) ** 0.5  # high dimensionality fix
        assert W.shape == (self.inputs, number), "W must be size [inputs, neurons] (expected [%d,%d])" % (self.inputs, number)
        assert B.shape == (number,), "B must be size [neurons] (expected [%d])" % number

        # add to an existing neuron type
        if func in self.neurons.keys():
            nn0, W0, B0 = self.neurons[func]
            number = nn0 + number
            W = np.hstack((W0, W))
            B = np.hstack((B0, B))

        self.neurons[func] = (number, W, B)

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


























