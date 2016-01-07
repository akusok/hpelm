# -*- coding: utf-8 -*-
"""Single hidden Layer Feed-forward neural Network.

SLFN neural network, which is trained by ELM algorithm. Stores parameters of a
network, keeps and operates neurons, saves and loads Neural Network model.
Keeps a connection with the `solver` object which implements fast computational math.
"""

import numpy as np
from tables import open_file
import cPickle
import os
import platform
from solver.slfn import Solver


class SLFN(object):
    """Initializes a SLFN with an empty hidden layer.
    
    Creates a Single Layer Feed-forward neural Network (SLFN). An ELM itself
    is a training algorithm for SLFN. The network needs to know a dimensionality
    of its `inputs` and `targets` (outputs), don't confuse with the number
    of data samples! It is fixed for the given SLFN/ELM model and never changes.
    It also takes a preferred `batch` size and a type of `accelerator`, 
    but they can be changed later.

    Parameters
    ----------
    inputs : int
        A dimensionality of input data, or a number of data features. Stays
        constant for a given SLFN and ELM model.
    targets : int
        A dimensionality of target data; also a number of classes in multi-class
        problems. Stays constant.
    batch : int, optional
        Dataset is processed in `batch`-size chunks, reducing memory requirements.
        Especially important for GPUs. A `batch` size less that 100 slows down
        processing speed.
    accelerator : {'GPU', 'SKCuda'}, optional
        An accelerator card to speed up computations, like Nvidia Tesla. The
        corresponding program must be installed in the system. 'SKCuda' and
        'Numba' are Python libraries and don't require compiling HPELM code,
        while others require to compile the corresponding code in 'hpelm/acc'
        folder. P.S. don't expect speedup on low-power laptop cards (needs
        at least GTX-series); also majority of cards speedup only single
        precision (32-bit float) computations --- exceptions are Nvidia Tesla
        series, Nvidia Titan/Titan Black (not Titan X), AMD FirePro s9000
        and up, Mac Pro with medium- and high-tier GPUs, and Intel Xeon Phi
        accelerator card.
        
    Raises
    ------
    AssertionError
        If you have wrong data dimensionality (see `inputs` and `targets`).
        
    Notes
    -----
    SLFN is created without any neurons, which are added later.
    
    Multiple types of neurons may be used.
    
    Neurons of one type may be added multiple times; they are joined
    together by the model for better performance.

    Example
    -------
    Example of creating an SLFN and adding neurons if you want to classify
    Iris dataset with 4 inputs and 3 classes. Note that SLFN is not called
    directly, but it's subclass ELM or HPELM is called.

    >>> from hpelm import ELM, HPELM
    >>> model = ELM(4, 3)
    >>> model.add_neurons(5, 'sigm')
    
    """




    def __del__(self):
        """Close HDF5 files opened during HPELM usage.
        """
        if len(self.opened_hdf5) > 0:
            for h5 in self.opened_hdf5:
                h5.close()





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


























