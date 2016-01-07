 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
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
        if accelerator is None:  # double precision Numpy solver
            self.solver = SLFN(self.targets, precision=self.precision)
            # TODO: add advanced and GPU solvers

        # init other stuff
        self.opened_hdf5 = []
        self.classification = None  # c / wc / mc
        self.wc = None  # weighted classification weights
        self.tprint = tprint  # time intervals in seconds to report ETA


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

        assert len(self.neurons) > 0, "Add neurons to ELM before training it"
        X, T = self._checkdata(X, T)
        args = [a.upper() for a in args]  # make all arguments upper case

        # kind of "enumerators", try to use only inside that script
        MODELSELECTION = None  # V / CV / MCCV / LOO / None

        # reset parameters
        self.ranking = None
        self.kmax_op = None
        self.classification = None  # c / wc / mc
        self.weights_wc = None  # weigths for weighted classification

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
                assert "w" in kwargs.keys(), "Provide class weights for weighted classification"
                w = kwargs['w']
                assert len(w) == T.shape[1], "Number of class weights differs from number of target classes"
                self.wc = w
                self.classification = "wc"
            if a == "MC":
                assert self.targets > 1, "Classification targets must have 1 output per class"
                self.classification = "mc"
            # if a in ("A", "AD", "ADAPTIVE"):
            #     assert "batch" in kwargs.keys(), "Provide batch size for adaptive ELM model (batch)"
            #     batch = kwargs['batch']
            #     ADAPTIVE = True

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
        """Update HH, HT matrices with training data (X,T).

        Performs balanced classification if self.classification="cb".
        """
        # initialize batch size
        nb = int(np.ceil(float(X.shape[0]) / self.batch))
        wc_vector = None

        # run scripts
        if self.classification == "wc" and self.wc is None:  # find weights automatically
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
        """
        if self.ranking == "OP":
            if self.kmax_op is None:  # set maximum number of neurons
                self.kmax_op = nn
            else:  # or set a limited number of neurons
                nn = self.kmax_op
            if T.shape[1] < 10:  # fast mrsr for less outputs but O(2^t) in outputs
                rank = mrsr(H, T, self.kmax_op)
            else:  # slow mrsr for many outputs but O(t) in outputs
                rank = mrsr2(H, T, self.kmax_op)
        else:
            rank, nn = super(ELM, self)._ranking(nn)
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
        """Add neurons to the SLFN.

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






















