 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from slfn import SLFN
from hpelm.modules import mrsr, mrsr2
from mss_v import train_v
from mss_cv import train_cv
from mss_loo import train_loo


class ELM(SLFN):
    """Interface for training Extreme Learning Machines.
    """

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
        ADAPTIVE = False  # batch / None

        # reset parameters
        self.ranking = None
        self.kmax_op = None
        self.classification = None  # c / wc / mc
        self.weights_wc = None  # weigths for weighted classification

        # check exclusive parameters
        assert len(set(args).intersection(set(["V", "CV", "LOO"]))) <= 1, "Use only one of V / CV / LOO"
        assert len(set(args).intersection(set(["C", "WC", "MC"]))) <= 1, "Use only one of \
            C (classification) / MC (multiclass) / WC (weighted classification)"

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
                self.weights_wc = w
                self.classification = "wc"
            if a == "MC":
                assert self.targets > 1, "Classification targets must have 1 output per class"
                self.classification = "mc"
            # if a in ("A", "AD", "ADAPTIVE"):
            #     assert "batch" in kwargs.keys(), "Provide batch size for adaptive ELM model (batch)"
            #     batch = kwargs['batch']
            #     ADAPTIVE = True

        # use "train_x" method which borrows _project(), _error() from the "self" object
        if MODELSELECTION == "V":
            train_v(self, X, T, Xv, Tv)
        elif MODELSELECTION == "CV":
            train_cv(self, X, T, k)
        elif MODELSELECTION == "LOO":
            train_loo(self, X, T)
        else:
            self._train(X, T)

    def _train(self, X, T):
        """Most basic training algorithm for an ELM.
        """
        self.Beta = self._project(X, T, solve=True)[2]

    def _project(self, X, T, solve=False):
        """Create HH, HT matrices and computes solution Beta.

        An ELM-specific projection for all usage cases.
        Returns solution Beta if solve=True.
        Runs on GPU if self.accelerator="GPU".
        Performs balanced classification if self.classification="cb".
        """
        # initialize
        nn = sum([n1[1] for n1 in self.neurons])
        batch = max(self.batch, nn)
        if X.shape[0] % batch > 0:
            nb = X.shape[0]/batch + 1
        else:
            nb = X.shape[0]/batch

        # GPU script
        def proj_gpu(self, X, T, getBeta, nn, nb):
            s = self.magma_solver.GPUSolver(nn, self.targets, self.alpha)
            for X0, T0 in zip(np.array_split(X, nb, axis=0),
                              np.array_split(T, nb, axis=0)):
                H0 = self.project(X0)
                s.add_data(H0, T0)
            HH, HT = s.get_corr()
            if getBeta:
                Beta = s.solve()
            else:
                Beta = None
            return HH, HT, Beta

        # CPU script
        def proj_cpu(self, X, T, getBeta, nn, nb):
            HH = np.zeros((nn, nn))
            HT = np.zeros((nn, self.targets))
            HH.ravel()[::nn+1] += self.alpha  # add to matrix diagonal trick
            for X0, T0 in zip(np.array_split(X, nb, axis=0),
                              np.array_split(T, nb, axis=0)):
                H0 = self.project(X0)
                HH += np.dot(H0.T, H0)
                HT += np.dot(H0.T, T0)
            if getBeta:
                Beta = self._solve_corr(HH, HT)
            else:
                Beta = None
            return HH, HT, Beta

        # run scripts
        if self.classification == "cb":  # balanced classification wrapper
            ns = T.sum(axis=0).astype(np.float64)  # number of samples in classes
            wc = (ns / ns.sum())**-1  # weights of classes
            HH = np.zeros((nn, nn))  # init data holders
            HT = np.zeros((nn, self.targets))
            for i in range(wc.shape[0]):  # iterate over each particular class
                idxc = T[:, i] == 1
                Xc = X[idxc]
                Tc = T[idxc]
                if self.accelerator == "GPU":
                    HHc, HTc, _ = proj_gpu(self, Xc, Tc, False, nn, nb)
                else:
                    HHc, HTc, _ = proj_cpu(self, Xc, Tc, False, nn, nb)
                HH += HHc * wc[i]
                HT += HTc * wc[i]
            if solve:  # obtain solution
                Beta = self._solve_corr(HH, HT)
        else:
            if self.accelerator == "GPU":
                HH, HT, Beta = proj_gpu(self, X, T, solve, nn, nb)
            else:
                HH, HT, Beta = proj_cpu(self, X, T, solve, nn, nb)

        # return results
        if solve:
            return HH, HT, Beta
        else:
            return HH, HT

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
                err = np.mean(errc * self.weights_wc)
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


























