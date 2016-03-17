# -*- coding: utf-8 -*-
"""This is a fast Python implementation of SLFN.

Created on Sun Sep  6 11:18:55 2015
@author: akusok
"""

import os
import platform

from .slfn import SLFN
import numpy as np
from scipy.linalg import blas, lapack


class SLFNPython(SLFN):
    """Single Layer Feed-forward Network (SLFN), the neural network that ELM trains.
    """

    def __init__(self, inputs, outputs, norm=None, precision=np.float64):

        super(SLFNPython, self).__init__(inputs, outputs, norm, precision)
        # get correct BLAS/LAPACK functions for precision
        if precision is np.float32:
            self.syrk = blas.ssyrk
            self.posv = lapack.sposv
        elif precision is np.float64:
            self.syrk = blas.dsyrk
            self.posv = lapack.dposv


    def add_batch(self, X, T, wc=None):
        """Add a batch using Symmetric Rank-K matrix update for HH.
        """
        H = self._project(X)
        T = T.astype(self.precision)
        if wc is not None:  # apply weights if given
            w = np.array(wc**0.5, dtype=self.precision)[:, None]  # re-shape to column matrix
            H *= w
            T *= w

        if self.HH is None:  # initialize space for self.HH, self.HT
            self.HH = np.zeros((self.L, self.L), dtype=self.precision)
            self.HT = np.zeros((self.L, self.outputs), dtype=self.precision)
            np.fill_diagonal(self.HH, self.norm)

        #self.syrk(1, H.T, 1, self.HH, trans=0, overwrite_c=1)  # 'overwrite_c' does not work
        self.HH = self.syrk(1, H.T, 1, self.HH, trans=0)  # self.HH += np.dot(H.T, H)
        self.HT += np.dot(H.T, T)

    def solve_corr(self, HH, HT):
        """Compute output weights B for given HH and HT.

        Simple but inefficient version, see a better one in solver_python.

        Args:
            HH (matrix): covariance matrix of hidden layer represenation H, size (`L` * `L`)
            HT (matrix): correlation matrix between H and outputs T, size (`L` * `outputs`)
        """
        _, B, info = self.posv(HH, HT)
        if info > 0:
            print("Covariance matrix is not full rank; solving with SVD (slow)")
            print("This happened because you have duplicated or too many neurons")
            HH = HH + np.triu(HH, k=1).T
            B = np.linalg.lstsq(HH, HT)[0]
        return B


    def get_corr(self):
        """Return current correlation matrices.
        """
        HH = self.HH + np.triu(self.HH, k=1).T
        return HH, self.HT
