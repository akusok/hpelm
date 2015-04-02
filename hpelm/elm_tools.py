# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from scipy.linalg import solve as cpu_solve

from slfn import SLFN
from modules import mrsr, mrsr2


class ELM_TOOLS(SLFN):
    """Non-parallel Extreme Learning Machine.
    """

    def __init__(self, inputs, outputs, accelerator="", batch=0):
        """Contructor of a basic ELM model.
        """
        super(ELM_TOOLS, self).__init__(inputs, outputs)
        self.norm = 1E-9  # normalization for H'H solution
        self.classification = None
        self.ranking = None
        self.kmax = None  # maximum number of neurons for OP-ELM
        self.accelerator = None  # None, "GPU", "PHI"
        self.batch = None
        if accelerator == "GPU":
            self.accelerator = "GPU"
            self.magma_solver = __import__('gpu.magma_solver', globals(), locals(), ['gpu_solve'], -1)
            self.batch = max(100, batch)
            print "using GPU"

    def _project(self, X, T, solve=False):
        """Create HH, HT matrices and computes solution Beta.

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
            s = self.magma_solver.GPUSolver(nn, self.targets, self.norm)
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
            HH.ravel()[::nn+1] += self.norm  # add to matrix diagonal trick
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

    def _solve_corr(self, HH, HT):
        """Solve a linear system from correlation matrices.
        """
        if self.accelerator == "GPU":
            Beta = self.magma_solver.gpu_solve(HH, HT, self.norm)
        else:
            Beta = cpu_solve(HH, HT, sym_pos=True)
        return Beta

    def _solve(self, H, T):
        """Solve a linear system.
        """
        Beta = cpu_solve(H, T)
        return Beta

    def _train(self, X, T):
        """Most basic training algorithm for an ELM.
        """
        self.Beta = self._project(X, T, solve=True)[2]

    def _error(self, Y, T, R=None):
        """Returns regression/classification/multiclass error, also for PRESS.
        """
        if R is None:  # normal classification error
            if self.classification == "c":
                err = (Y.argmax(1) != T.argmax(1))
            elif self.classification == "mc":
                err = ((Y > 0.5) - (T > 0.5))
            else:
                err = Y - T
        else:  # LOO_PRESS error
            if self.classification == "c":
                err = (Y.argmax(1) != T.argmax(1)) / R.ravel()
            elif self.classification == "mc":
                err = ((Y > 0.5) - (T > 0.5)) / R.reshape((-1, 1))
            else:
                err = (Y - T) / R.reshape((-1, 1))
        nerr = np.mean(err**2)  # get 2-norm
        return nerr

    def _ranking(self, nn, H=None, T=None):
        """Return ranking of hidden neurons; random or OP.
        """
        if self.ranking is "OP":
            if self.kmax is None:  # set maximum number of neurons
                self.kmax = nn
            else:  # or set a limited number of neurons
                nn = self.kmax
            if T.shape[1] > 10:  # fast mrsr for less outputs but O(2^t) in outputs
                rank = mrsr(H, T, self.kmax)
            else:  # slow mrsr for many outputs but O(t) in outputs
                rank = mrsr2(H, T, self.kmax)
        else:
            rank = np.arange(nn)
            np.random.shuffle(rank)
        return rank, nn

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




















