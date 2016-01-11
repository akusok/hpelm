 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import cPickle
from tables import open_file
from nnets.slfn import SLFN
from hpelm.modules import mrsr, mrsr2
from mss_v import train_v
from mss_cv import train_cv
from mss_loo import train_loo


class ELM(object):
    """Interface for training Extreme Learning Machines (ELM).

    Args:
        inputs (int): dimensionality of input data, or number of data features
        outputs (int): dimensionality of output data, or number of classes
        batch (int, optional): batch size for data processing in ELM, reduces memory requirements. Does not work
            for model structure selection (validation, cross-validation, Leave-One-Out). Can be changed later directly
            as a class attribute.
        accelerator (string, optional): type of accelerated ELM to use: None, 'GPU', ...
        precision (optional): data precision to use, supports signle ('single', '32' or numpy.float32) or double
            ('double', '64' or numpy.float64). Single precision is faster but may cause numerical errors. Majority
            of GPUs work in single precision. Default: **double**.
        norm (double, optinal): L2-normalization parameter, None gives the default value.
        tprint (int, optional): ELM reports its progess every `tprint` seconds or after every batch, whatever takes longer.

    Class attributes; attributes that simply store initialization or `train()` parameters are omitted.

    Attributes:
        nnet (object): Implementation of neural network with computational methods, but without
            complex logic. Different implementations are given by different classes: for Python, for GPU, etc.
            See ``hpelm.nnets`` folder for particular files. You can implement your own computational algorithm
            by inheriting from ``hpelm.nnets.SLFN`` and overwriting some methods.
        flist (list of strings): Awailable types of neurons, use them when adding new neurons.

    Note:
        Below the 'matrix' type means a 2-dimensional Numpy.ndarray.
    """

    def __init__(self, inputs, outputs, batch=1000, accelerator=None, precision='double', norm=None, tprint=5):
        assert isinstance(inputs, (int, long)), "Number of inputs must be integer"
        assert isinstance(outputs, (int, long)), "Number of outputs must be integer"
        assert batch > 0, "Batch should be positive"

        self.inputs = inputs
        self.outputs = outputs
        self.batch = int(batch)
        self.precision = np.float64

        if precision in (np.float32, np.float64):
            self.precision = precision
        elif 'double' in precision.lower() or '64' in precision:
            self.precision = np.float64
        elif 'single' in precision or '32' in precision:
            self.precision = np.float32
        else:
            print "Unknown precision parameter: %s, using double precision" % precision

        # create SLFN solver to do actual computations
        self.accelerator = accelerator
        if accelerator is None:  # double precision Numpy solver
            self.nnet = SLFN(self.outputs, precision=self.precision, norm=norm)
            # TODO: add advanced and GPU nnets, in load also

        # init other stuff
        self.opened_hdf5 = []  # list of opened HDF5 files, they are closed in ELM descructor
        self.classification = None  # c / wc / mc
        self.wc = None  # weighted classification weights
        self.tprint = tprint  # time intervals in seconds to report ETA
        self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf")  # supported neuron types

    def __str__(self):
        s = "ELM with %d inputs and %d outputs\n" % (self.inputs, self.outputs)
        s += "Hidden layer neurons: "
        for n, func, _, _ in self.nnet.neurons:
            s += "%d %s, " % (n, func)
        s = s[:-2]
        return s

    def train(self, X, T, *args, **kwargs):
        """Universal training interface for ELM model with model structure selection.

        Model structure selection takes more time and requires all data to fit into memory. Optimal pruning ('OP',
        effectively an L1-regularization) takes the most time but gives the smallest and best performing model.
        Choosing a classification forces ELM to use classification error in model structure selection,
        and in `error()` method output.

        Args:
            X (matrix): input data matrix, size (N * `inputs`)
            T (matrix): outputs data matrix, size (N * `outputs`)
            'c'/'wc'/'mc' (string, choose one): train ELM for classification ('c'), classification with weighted classes
                ('wc') or multi-label classification ('mc') with several correct classes per data sample. In classification,
                number of `outputs` is the number of classes; correct class(es) for each sample has value 1 and
                incorrect classes have 0.
            'V'/'CV'/'LOO' (sting, choose one): model structure selection: select optimal number of neurons using a validation set ('V'),
                cross-validation ('CV') or Leave-One-Out ('LOO')
            'OP' (string, use with 'V'/'CV'/'LOO'): choose best neurons instead of random ones, training takes longer; equivalent to L1-regularization

        Keyword Args:
            Xv (matrix, use with 'V'): validation set input data, size (Nv * `inputs`)
            Tv (matrix, use with 'V'): validation set outputs data, size (Nv * `outputs`)
            k (int, use with 'CV'): number of splits for cross-validation, k>=3
            kmax (int, optional, use with 'OP'): maximum number of neurons to keep in ELM
            batch (int, optional): batch size for ELM, overwrites batch size from an initialization
        """

        assert len(self.nnet.neurons) > 0, "Add neurons to ELM before training it"
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
        # TODO: code coverage for model structure selection
        for a in args:
            if a == "V":  # validation set
                assert "Xv" in kwargs.keys(), "Provide validation dataset (Xv)"
                assert "Tv" in kwargs.keys(), "Provide validation outputs (Tv)"
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
                assert self.outputs > 1, "Classification outputs must have 1 output per class"
                self.classification = "c"
            if a == "WC":
                assert self.outputs > 1, "Classification outputs must have 1 output per class"
                self.classification = "wc"
                if 'w' in kwargs.keys():
                    w = np.array(kwargs['w'])
                    assert len(w) == T.shape[1], "Number of class weights must be equal to the number of classes"
                    self.wc = w
            if a == "MC":
                assert self.outputs > 1, "Classification outputs must have 1 output per class"
                self.classification = "mc"
            # TODO: Adaptive ELM model for timeseries (someday)

        if "batch" in kwargs.keys():
            self.batch = int(kwargs["batch"])

        self.nnet.reset()  # remove previous training
        # use "train_x" method which borrows _project(), _error() from the "self" object
        if MODELSELECTION == "V":
            train_v(self, X, T, Xv, Tv)
        elif MODELSELECTION == "CV":
            train_cv(self, X, T, k)
        elif MODELSELECTION == "LOO":
            train_loo(self, X, T)
        else:
            # basic training algorithm
            self.add_data(X, T)
            self.nnet.solve()

    def add_data(self, X, T):
        """Feed new training data (X,T) to ELM model in batches; does not solve ELM itself.

        Helper method that updates intermediate solution parameters HH and HT, which are used for solving ELM later.
        Updates accumulate, so this method can be called multiple times with different parts of training data.
        To reset accumulated training data, use `ELM.nnet.reset()`.

        For training an ELM use `ELM.train()` instead.

        Args:
            X (matrix): input data
            T (matrix): output data
        """
        # initialize batch size
        nb = int(np.ceil(float(X.shape[0]) / self.batch))
        wc_vector = None

        # find automatic weights if none are given
        if self.classification == "wc" and self.wc is None:
            ns = T.sum(axis=0).astype(self.nnet.precision)  # number of samples in classes
            self.wc = (ns / ns.sum())**-1  # weights of classes

        for X0, T0 in zip(np.array_split(X, nb, axis=0),
                          np.array_split(T, nb, axis=0)):
            if self.classification == "wc":
                wc_vector = self.wc[np.where(T0 == 1)[1]]  # weights for samples in the batch
            self.nnet.add_batch(X0, T0, wc_vector)

    def add_neurons(self, number, func, W=None, B=None):
        """Adds neurons to ELM model. ELM is created empty, and needs some neurons to work.

        Add neurons to an empty ELM model, or add more neurons to a model that already has some.

        Random weights `W` and biases `B` are generated automatically if not provided explicitly.
        Maximum number of neurons is limited by the available RAM and computational power, a sensible limit
        would be 1000 neurons for an average size dataset and 15000 for the largest datasets. ELM becomes slower after
        3000 neurons because computational complexity is proportional to a qube of number of neurons.

        This method checks and prepares neurons, they are actually stored in `solver` object.

        Args:
            number (int): number of neurons to add
            func (string): type of neurons: "lin" for linear, "sigm" or "tanh" for non-linear,
                "rbf_l1", "rbf_l2" or "rbf_linf" for radial basis function neurons.
            W (matrix, optional): random projection matrix size (`inputs` * `number`). For 'rbf_' neurons,
                W stores centroids of radial basis functions in transposed form.
            B (vector, optional): bias vector of size (`number` * 1), a 1-dimensional Numpy.ndarray object.
                For 'rbf_' neurons, B gives widths of radial basis functions.
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
        self.nnet.add_neurons(number, func, W, B)

    def error(self, Y, T):
        """Calculate error of model predictions.

        Computes Mean Squared Error (MSE) between model predictions Y and true outputs T.
        For classification, computes corresponding classification error. For weighted classification,
        computes average weighted per-class error. For multi-label classification, correct classes are all with Y>0.5.

        Args:
            Y (matrix): ELM model predictions, can be computed with `predict()` function.
            T (matrix): true outputs.

        Returns:
            e (double): MSE for regression / classification error for classification.
        """
        _, Y = self._checkdata(None, Y)
        _, T = self._checkdata(None, T)
        return self._error(Y, T)

    def confusion(self, Y1, T1):
        """Unfinished...

        Args:
            Y1 (matrix): predicted outputs by ELM model, size (N * `outputs`)
            T1 (matrix): true outputs or classes, size (N * `outputs`)

        Returns:
            conf (matrix): confusion matrix, size (`outputs` * `outputs`)
        """
        # TODO: Fix confusion matrix code
        _, Y = self._checkdata(None, Y1)
        _, T = self._checkdata(None, T1)
        nn = np.sum([n1[1] for n1 in self.nnet.neurons])
        N = T.shape[0]
        batch = max(self.batch, nn)
        nb = int(np.ceil(float(N) / self.batch))  # number of batches

        C = self.outputs
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

    def project(self, X):
        """Get ELM's hidden layer representation of input data.

        Args:
            X (matrix): input data, size (N * `inputs`)

        Returns:
            H (matrix): hidden layer representation matrix, size (N * number_of_neurons)
        """
        X, _ = self._checkdata(X, None)
        H = self.nnet.project(X)
        return H

    def predict(self, X):
        """Predict outputs Y for the given input data X.

        Args:
            X (matrix): input data of size (N * `inputs`)

        Returns:
            Y (matrix): output data or predicted classes, size (N * `outputs`).
        """
        X, _ = self._checkdata(X, None)
        Y = self.nnet.predict(X)
        return Y

    def save(self, fname):
        """Save ELM model with current parameters.

        Model does not save a particular solver, precision batch size. They are obtained from
        a new ELM when loading the model (so one can switch to another solver, for instance).

        Also ranking and max number of neurons are not saved, because they
        are runtime training info irrelevant after the training completes.

        Args:
            fname (string): filename to save model into.
        """
        assert isinstance(fname, basestring), "Model file name must be a string"
        m = {"inputs": self.inputs,
             "outputs": self.outputs,
             "Classification": self.classification,
             "Weights_WC": self.wc,
             "neurons": self.nnet.neurons,
             "norm": self.nnet.norm,  # W and bias are here
             "Beta": self.nnet.get_B()}
        try:
            cPickle.dump(m, open(fname, "wb"), -1)
        except IOError:
            raise IOError("Cannot create a model file at: %s" % fname)

    def load(self, fname):
        """Load ELM model data from a file.

        Load requires an ``ELM`` object, and it uses solver type, precision and batch size from that ELM object.

        Args:
            fname (string): filename to load model from.
        """
        assert isinstance(fname, basestring), "Model file name must be a string"
        try:
            m = cPickle.load(open(fname, "rb"))
        except IOError:
            raise IOError("Model file not found: %s" % fname)
        self.inputs = m["inputs"]
        self.outputs = m["outputs"]
        self.classification = m["Classification"]
        self.wc = m["Weights_WC"]

        # create a new solver and load neurons / Beta into it with correct precision
        if self.accelerator is None:
            self.nnet = SLFN(self.outputs, precision=self.precision)
        for number, func, W, B in m["neurons"]:
            self.nnet.add_neurons(number, func, W.astype(self.precision), B.astype(self.precision))
        self.nnet.norm = m["norm"]
        self.nnet.set_B(np.array(m["Beta"], dtype=self.precision))

    def __del__(self):
        # Closes any HDF5 files opened during HPELM usage.
        for h5 in self.opened_hdf5:
            h5.close()

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
        :param T: outputs matrix needed for optimal pruning
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
            msg = "T has wrong dimensionality: expected %d, found %d" % (self.outputs, T.shape[1])
            assert T.shape[1] == self.outputs, msg

        if (X is not None) and (T is not None):
            assert X.shape[0] == T.shape[0], "X and T cannot have different number of samples"

        return X, T










