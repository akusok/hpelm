 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import cPickle
from tables import open_file
from solvers.slfn import SLFN
from hpelm.modules import mrsr, mrsr2
from mss_v import train_v
from mss_cv import train_cv
from mss_loo import train_loo


class ELM(object):
    """Interface for training Extreme Learning Machines.
    """

    def __init__(self, inputs, targets, batch=1000, accelerator=None, precision='single', tprint=5):
        """Single-hidden Layer Feed-forward Network.
        """

        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(targets, (int, long)), "Number of outputs must be integer"
        assert batch > 0, "Batch should be positive"

        self.inputs = inputs
        self.targets = targets
        self.batch = int(batch)
        self.precision = np.float32

        if 'double' in precision.lower() or '64' in precision or precision is np.float64:
            self.precision = np.float64
        elif 'single' in precision or '32' in precision or precision is np.float32:
            self.precision = np.float32
        else:
            print "Unknown precision parameter: %s, using single precision" % precision

        # create SLFN solver to do actual computations
        self.accelerator = accelerator
        if accelerator is None:  # double precision Numpy solver
            self.solver = SLFN(self.targets, precision=self.precision)
            # TODO: add advanced and GPU solvers, in load also

        # init other stuff
        self.opened_hdf5 = []
        self.classification = None  # c / wc / mc
        self.wc = None  # weighted classification weights
        self.tprint = tprint  # time intervals in seconds to report ETA
        self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")  # supported neuron types

    def __del__(self):
        """Close any HDF5 files opened during HPELM usage.
        """
        for h5 in self.opened_hdf5:
            h5.close()

    def __str__(self):
        s = "ELM with %d inputs and %d outputs\n" % (self.inputs, self.targets)
        s += "Hidden layer neurons: "
        for func, n, _, _ in self.solver.neurons:
            s += "%d %s, " % (n, func)
        s = s[:-2]
        return s

    def train(self, X, T, *args, **kwargs):
        """Universal training interface for ELM model with model structure selection.

        :param X: input data matrix
        :param T: target data matrix

        Model structure selection (exclusive, choose one)
        :param "V": use validation set
        :param "CV": use cross-validation
        :param "LOO": use leave-one-out validation

        Additional parameters for model structure selecation
        :param Xv: validation data X ("V")
        :param Tv: validation targets T ("V")
        :param k: number of splits ("CV")

        Ranking of hidden neurons
        :param "OP": use Optimal Pruning (OP-ELM)
        :param "kmax": maximum number of neurons (with "OP")

        System setup
        :param "classification"/"c": build ELM for classification
        :param "weighted classification"/"wc": build ELM with weights assigned to classes
        :param w: weights of classes for "wc"
        :param "multiclass"/"mc": build ELM for multiclass classification
        :param "adaptive"/"ad": build adaptive ELM for non-stationary model
        :param "batch": batch size for adaptive ELM (sliding window step size)
        """

        assert len(self.solver.neurons) > 0, "Add neurons to ELM before training it"
        X, T = self._checkdata(X, T)
        args = [a.upper() for a in args]  # make all arguments upper case

        # kind of "enumerators", try to use only inside that script
        MODELSELECTION = None  # V / CV / MCCV / LOO / None

        # reset parameters
        self.ranking = None
        self.kmax_op = None
        self.classification = None  # c / wc / mc
        self.wc = None  # weigths for weighted classification

        # check exclusive parameters
        assert len(set(args).intersection(set(["V", "CV", "LOO"]))) <= 1, "Use only one of V / CV / LOO"
        msg = "Use only one of: C (classification) / MC (multiclass) / WC (weighted classification)"
        assert len(set(args).intersection(set(["C", "WC", "MC"]))) <= 1, msg

        # parse parameters
        for a in args:
            if a == "V":  # validation set
                assert "Xv" in kwargs.keys(), "Provide validation dataset (Xv)"
                assert "Tv" in kwargs.keys(), "Provide validation targets (Tv)"
                Xv = kwargs['Xv']
                Tv = kwargs['Tv']
                Xv, Tv = self._checkdata(Xv, Tv)
                MODELSELECTION = "V"
            if a == "CV":
                assert "k" in kwargs.keys(), "Provide Cross-Validation number of splits (k)"
                k = kwargs['k']
                assert k >= 3, "Use at least k=3 splits for Cross-Validation"
                MODELSELECTION = "CV"
            if a == "LOO":
                MODELSELECTION = "LOO"
            if a == "OP":
                self.ranking = "OP"
                if "kmax" in kwargs.keys():
                    self.kmax_op = int(kwargs["kmax"])
            if a == "C":
                assert self.targets > 1, "Classification targets must have 1 output per class"
                self.classification = "c"
            if a == "WC":
                assert self.targets > 1, "Classification targets must have 1 output per class"
                self.classification = "wc"
                if 'w' in kwargs.keys():
                    w = kwargs['w']
                    assert len(w) == T.shape[1], "Number of class weights must be equal to the number of classes"
                    self.wc = w
            if a == "MC":
                assert self.targets > 1, "Classification targets must have 1 output per class"
                self.classification = "mc"
            # TODO: Adaptive ELM model for timeseries (someday)

        self.solver.reset()  # remove previous training
        # use "train_x" method which borrows _project(), _error() from the "self" object
        if MODELSELECTION == "V":
            train_v(self, X, T, Xv, Tv)
        elif MODELSELECTION == "CV":
            train_cv(self, X, T, k)
        elif MODELSELECTION == "LOO":
            train_loo(self, X, T)
        else:
            # basic training algorithm
            self.add_batch(X, T)
            self.solver.solve()

    def add_batch(self, X, T):
        """Update HH, HT matrices using parts of training data (X,T).
        """
        # initialize batch size
        nb = int(np.ceil(float(X.shape[0]) / self.batch))
        wc_vector = None

        # find automatic weights if none are given
        if self.classification == "wc" and self.wc is None:
            ns = T.sum(axis=0).astype(self.solver.precision)  # number of samples in classes
            self.wc = (ns / ns.sum())**-1  # weights of classes

        for X0, T0 in zip(np.array_split(X, nb, axis=0),
                          np.array_split(T, nb, axis=0)):
            if self.classification == "wc":
                wc_vector = self.wc[np.where(T0 == 1)[1]]  # weights for samples in the batch
            self.solver.add_batch(X0, T0, wc_vector)

    def _error(self, Y, T, R=None):
        """Returns regression/classification/multiclass error, also for PRESS.

        An ELM-specific error with PRESS support.
        """
        if R is None:  # normal classification error
            if self.classification == "c":
                err = np.mean(Y.argmax(1) != T.argmax(1))
            elif self.classification == "wc":  # weighted classification
                c = T.shape[1]
                errc = np.zeros(c)
                for i in xrange(c):  # per-class MSE
                    idx = np.where(T[:, i] == 1)[0]
                    if len(idx) > 0:
                        errc[i] = np.mean(Y[idx].argmax(1) != i)
                err = np.mean(errc * self.wc)
            elif self.classification == "mc":
                err = np.mean((Y > 0.5) != (T > 0.5))
            else:
                err = np.mean((Y - T)**2)
        else:  # LOO_PRESS error
            if self.classification == "c":
                err = (Y.argmax(1) != T.argmax(1)).astype(np.float) / R.ravel()
                err = np.mean(err**2)
            elif self.classification == "wc":  # balanced classification
                c = T.shape[1]
                errc = np.zeros(c)
                for i in xrange(c):  # per-class MSE
                    idx = np.where(T[:, i] == 1)[0]
                    if len(idx) > 0:
                        t = (Y[idx].argmax(1) != i).astype(np.float) / R[idx].ravel()
                        errc[i] = np.mean(t**2)
                err = np.mean(errc * self.weights_wc)
            elif self.classification == "mc":
                err = ((Y > 0.5) != (T > 0.5)).astype(np.float) / R.reshape((-1, 1))
                err = np.mean(err**2)
            else:
                err = (Y - T) / R.reshape((-1, 1))
                err = np.mean(err**2)
        assert not np.isnan(err), "Error is NaN at %s" % self.classification
        return err

    def _ranking(self, nn, H=None, T=None):
        """Return ranking of hidden neurons; random or OP.

        :param nn: number of neurons
        :param H: data matrix needed for optimal pruning
        :param T: targets matrix needed for optimal pruning
        """
        if self.ranking == "OP":  # optimal ranking (L1 normalization)
            assert H is not None and T is not None, "Need H and T to perform optimal pruning"
            if self.kmax_op is not None:  # apply maximum number of neurons
                nn = self.kmax_op
            if T.shape[1] < 10:  # fast mrsr for less outputs but O(2^t) in outputs
                rank = mrsr(H, T, nn)
            else:  # slow mrsr for many outputs but O(t) in outputs
                rank = mrsr2(H, T, nn)
        else:  # random ranking
            rank = np.arange(nn)
            np.random.shuffle(rank)
        return rank, nn

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
        """Check and prepare neurons here, then pass them to SLFN.
        """
        assert isinstance(number, int), "Number of neurons must be integer"
        assert (func in self.flist or isinstance(func, np.ufunc)),\
            "'%s' neurons not suppored: use a standard neuron function or a custom <numpy.ufunc>" % func
        assert isinstance(W, (np.ndarray, type(None))), "Projection matrix (W) must be a Numpy ndarray"
        assert isinstance(B, (np.ndarray, type(None))), "Bias vector (B) must be a Numpy ndarray"

        # default neuron initializer
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
        # set to correct precision
        W = W.astype(self.precision)
        B = B.astype(self.precision)

        # add prepared neurons to the model
        self.solver.add_neurons(number, func, W, B)

    def save(self, fname):
        """Save ELM model with current parameters.

        Model does not save type of solver, batch size and report interval,
        gets them from ELM initialization instead (so one can use another solver, for instance).

        Also ranking and max number of neurons are not saved, because they
        are runtime training info irrelevant after training completes.

        :param fname: file name to save model into.
        :return:
        """
        assert isinstance(fname, basestring), "Model file name must be a string"
        m = {"inputs": self.inputs,
             "outputs": self.targets,
             "precision": self.precision,
             "Classification": self.classification,
             "Weights_WC": self.wc,
             "neurons": self.solver.neurons,
             "norm": self.solver.norm,  # W and bias are here
             "Beta": self.solver.get_B()}
        try:
            cPickle.dump(m, open(fname, "wb"), -1)
        except IOError:
            raise IOError("Cannot create a model file at: %s" % fname)

    def load(self, fname):
        """Load model data from a file.

        :param fname: model file name.
        :return:
        """
        assert isinstance(fname, basestring), "Model file name must be a string"
        try:
            m = cPickle.load(open(fname, "rb"))
        except IOError:
            raise IOError("Model file not found: %s" % fname)
        self.inputs = m["inputs"]
        self.targets = m["outputs"]
        self.precision = m["precision"],
        self.classification = m["Classification"]
        self.wc = m["Weights_WC"]

        if self.accelerator is None:  # double precision Numpy solver
            self.solver = SLFN(self.targets, precision=self.precision)
        self.solver.neurons = m["neurons"]
        self.solver.L = sum([n[1] for n in self.solver.neurons])  # number of neurons
        self.solver.norm = m["norm"]
        self.solver.set_B(m["Beta"])


#############################################################
###  Methods interacting directly with SLFN model of ELM  ###

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
        assert self.solver.get_B() is not None, "Train ELM before predicting"
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
        # TODO: Fix confusion matrix code
        _, Y = self._checkdata(None, Y1)
        _, T = self._checkdata(None, T1)
        nn = np.sum([n1[1] for n1 in self.solver.neurons])
        N = T.shape[0]
        batch = max(self.batch, nn)
        nb = int(np.ceil(float(N) / self.batch))  # number of batches

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











