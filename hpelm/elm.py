# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit as sigm
import cPickle


class ELM_abstract(object):

    def __init__(self, inputs, outputs, kind=""):
        assert isinstance(inputs, (int, long))
        assert isinstance(outputs, (int, long))
        assert isinstance(kind, basestring)
        self.inputs = 0
        self.targets = 0
        self.neurons = {}
        self.Beta = None

    def save(self, model):
        assert isinstance(model, basestring)
        model = {"inputs": self.inputs,
                 "outputs": self.targets,
                 "neurons": self.neurons,
                 "Beta": self.Beta}

    def load(self, model):
        assert isinstance(model, basestring)

    def add_neurons(self, number, func, W=None, B=None):
        assert isinstance(number, int)
        assert isinstance(func, (basestring, np.ufunc))
        assert isinstance(W, np.ndarray)
        assert isinstance(B, np.ndarray)
        # here 'ufunc' is derived from neuron name
        nn = number
        C = W
        s = B
        ntype = {"lin": (nn, W, B),  # projection here
                 "sigm": (nn, W, B),
                 "tanh": (nn, W, B),
                 "rbf_l1": (nn, C, s),  # distances here
                 "rbf_l2": (nn, C, s),
                 "rbf_linf": (nn, C, s),
                 np.sin: (nn, W, B)}  # only projection for custom functions
        self.neurons[func] = ntype

    def _project(self, X):
        # projects X into H
        H = X
        return H

    def _checkdata(self, X, T):
        # checks X and T, loads them from file
        return X, T

    def train(self, X, T):
        pass

    def predict(self, X):
        pass


class ELM(ELM_abstract):
    """Non-parallel Extreme Learning Machine.
    """

    def __init__(self, inputs, outputs, kind=""):
        """Create ELM of desired kind.

        :param regression: type of ELM task, can be regression, classification or timeseries regression.
        :param sparse: set to create an ELM with sparse projection matrix.
        """
        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(outputs, (int, long)), "Number of outputs must be integer"
        assert isinstance(kind, basestring), "Kind of ELM must be a string"

        self.classification = False
        self.multiclass = False
        self.regression = False
        if "classification" in kind:
            self.classification = True
        elif "multiclass" in kind:
            self.multiclass = True
        else:
            self.regression = True

        # set default argument values
        self.inputs = inputs
        self.targets = outputs
        self.Beta = None
        self.neurons = {}  # list of all neuron types

        # settings
        self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")

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
        # check and fill input data
        assert isinstance(number, int), "Number of neurons must be integer"
        assert func in self.flist or isinstance(func, np.ufunc),\
            "Type standard neuron function or use custom <numpy.ufunc>"

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
        assert isinstance(W, np.ndarray), "W must be a numpy array"
        assert isinstance(B, np.ndarray), "B must be a numpy array"
        assert W.shape == (self.inputs, number), "W must be size [inputs, neurons]"
        assert B.shape == (number,), "B must be size [neurons]"

        # generate new neuron type
        if func in self.neurons.keys():
            nn0, W0, B0 = self.neurons[func]
            ntype = (nn0 + number, np.hstack((W0, W)), np.hstack((B0, B)))
        else:
            ntype = (number, W, B)

        # save new neuron type
        self.neurons[func] = ntype

    def save(self, model):
        assert isinstance(model, basestring), "Model file name must be a string"
        m = {}
        m["inputs"] = self.inputs
        m["outputs"] = self.targets
        m["neurons"] = self.neurons
        m["Beta"] = self.Beta
        try:
            cPickle.dump(m, open(model, "wb"), -1)
        except IOError:
            raise IOError("Cannot create a model file at: %s" % model)

    def load(self, model):
        assert isinstance(model, basestring), "Model file name must be a string"
        try:
            model = cPickle.load(open(model, "rb"))
        except IOError:
            raise IOError("Model file not found: %s" % model)
        self.inputs = model["inputs"]
        self.targets = model["outputs"]
        self.neurons = model["neurons"]
        self.Beta = model["Beta"]

    def _project(self, X):
        # assemble global hidden layer output
        H = []
        for func, ntype in self.neurons.iteritems():
            _, W, B = ntype

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
                np.exp(H0, out=H0)
            elif func == "sigm":
                sigm(H0, out=H0)
            elif func == "tanh":
                np.tanh(H0, out=H0)
            else:
                H0 = func(H0)  # custom <numpy.ufunc>
            H.append(H0)

        H = np.hstack(H)
        return H

    def _checkdata(self, X, T):
        # check input data
        if X is not None:
            assert isinstance(X, np.ndarray), "X must be a numpy array"
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            assert len(X.shape) == 2, "X must be 2-dimensional matrix"
            assert X.shape[1] == self.inputs, \
                "X has wrong dimensionality: expected %d, found %d" % (self.inputs, X.shape[1])

        if T is not None:
            assert isinstance(T, np.ndarray), "T must be a numpy array"
            if len(T.shape) == 1:
                T = T.reshape(-1, 1)
            assert len(X.shape) == 2, "T must be 1- or 2-dimensional matrix"
            assert T.shape[1] == self.targets, \
                "T has wrong dimensionality: expected %d, found %d" % (self.targets, T.shape[1])

        assert X.shape[0] == T.shape[0], "X and T must have the same number of samples"
        return X, T

    def train(self, X, T):
        """Learn a model to project inputs X to targets T.

        :param X: - matrix of inputs
        :param T: - matrix of targets
        """
        assert len(self.neurons) > 0, "Add neurons before training ELM"
        X, T = self._checkdata(X, T)
        H = self._project(X)
        self.Beta = np.linalg.pinv(H).dot(T)

    def predict(self, X):
        """Predict targets for the given inputs X.

        :param X: - model inputs
        """
        assert self.Beta is not None, "Train ELM before predicting"
        X, _ = self._checkdata(X, None)
        H = self._project(X)
        T_hat = H.dot(self.Beta)
        return T_hat
































































    '''  COPY OF OLD PRUNING METHODS
    def prune_op(self, X, T, batch=10000, delimiter=" "):
        """Prune ELM as in OP-ELM paper.
        """        
        # get data iterators
        genX, self.inputs, N = batchX(X, batch, delimiter)
        genT, self.targets  = batchT(T, batch, delimiter, self.C_dict)
        
        # project data
        nn = len(self.ufunc)
        delta = 0.95  # improvement of MSE for adding more neurons
        nfeats = []
        neurons = np.zeros((nn,))
        for X1,T1 in zip(genX, genT):
            
            # get hidden layer outputs
            H = np.dot(X1,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            H,T1 = semi_Tikhonov(H,T1,self.Tmean)  # add Tikhonov regularization
            
            # get ranking of neurons in that batch
            rank = mrsr(H, T1, nn)
            
            # select best number of neurons
            MSE = press(H[:, rank[:2]], T1, self.classification, self.multiclass)
            R_opt = rank[:2]
            early_stopping = int(nn/10) + 1  # early stopping if no improvement in 10% neurons
            last_improvement = 0
            for i in range(3, nn):
                last_improvement += 1
                r = rank[:i]
                mse1 = press(H[:,r], T1, self.classification, self.multiclass)
                if mse1 < MSE * delta:
                    MSE = mse1
                    R_opt = r
                    last_improvement = 0
                elif last_improvement > early_stopping:  # early stopping if MSE raises 
                    break
            r = R_opt
            
            # save number of neurons and their ranking information
            nfeats.append(len(r)) 
            # first selected neuron gets weight 2, last one gets weight 1
            neurons[r] += np.linspace(2,1,num=len(r))

        # combine neuron ranking
        nfeats = np.round(np.mean(nfeats))
        neurons = np.argsort(neurons)[::-1][:nfeats]  # sorting in descending order
        
        # update ELM parameters and re-calculate B
        self.W = self.W[:,neurons]
        self.ufunc = [self.ufunc[j] for j in neurons]
        self.train(X, T, batch=batch, delimiter=delimiter)
        


    def prune_op2(self, X, T, norm=1, batch=10000, delimiter=" "):
        """Prune ELM with a more recent implementation of MRSR.
        
        :param norm: - check numpy.linalg.norm(X, <norm>)
        """        
        # get data iterators
        genX, self.inputs, N = batchX(X, batch, delimiter)
        genT, self.targets  = batchT(T, batch, delimiter, self.C_dict)
        
        # project data
        nn = len(self.ufunc)
        delta = 0.95  # improvement of MSE for adding more neurons
        nfeats = []
        neurons = np.zeros((nn,))
        for X1,T1 in zip(genX, genT):
            
            # get hidden layer outputs
            H = np.dot(X1,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            H,T1 = semi_Tikhonov(H,T1,self.Tmean)  # add Tikhonov regularization

            # get ranking of neurons in that batch
            # this MRSR2 is a class, with <.rank> attribute and <.new_input()> method
            M = mrsr2(H, T1, norm)
            M.new_input()
            M.new_input()
            
            # select best number of neurons
            MSE = press(H[:, M.rank], T1, self.classification, self.multiclass)
            R_opt = M.rank
            early_stopping = int(nn/10) + 1  # early stopping if no improvement in 10% neurons
            last_improvement = 0
            for i in range(3, nn):
                last_improvement += 1
                M.new_input()
                mse1 = press(H[:, M.rank], T1, self.classification, self.multiclass)
                if mse1 < MSE * delta:
                    MSE = mse1
                    R_opt = M.rank
                    last_improvement = 0
                elif last_improvement > early_stopping:  # early stopping if MSE raises 
                    break
            rank = R_opt
            del M            
            
            # save number of neurons and their ranking information
            nfeats.append(len(rank)) 
            # first selected neuron gets weight 2, last one gets weight 1
            neurons[rank] += np.linspace(2,1,num=len(rank))

        # combine neuron ranking
        nfeats = np.round(np.mean(nfeats))
        neurons = np.argsort(neurons)[::-1][:nfeats]  # sorting in descending order
        
        # update ELM parameters and re-calculate B
        self.W = self.W[:,neurons]
        self.ufunc = [self.ufunc[j] for j in neurons]
        self.train(X, T, batch=batch, delimiter=delimiter)
    '''
































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        