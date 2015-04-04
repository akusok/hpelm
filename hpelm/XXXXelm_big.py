 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import lstsq

from modules.data_loader import batchX, batchT, meanstdX, c_dictT, decode
from modules.regularizations import semi_Tikhonov
from modules.error_functions import mse
from elm_abstract import ELM_abstract


class HPELM(ELM_abstract):
    """Extreme Learning Machine for Big Data.
    """

    # inherited  def add_neurons(self, number, func, W=None, B=None):
    # inherited  def save(self, model):
    # inherited  def load(self, model):
    # inherited  def _checkdata(self, X, T):

    def __init__(self, inputs, outputs, kind="", batch=10000):
        """Create ELM of desired kind.
        """
        super(HPELM, self).__init__(inputs, outputs, kind)
        self.batch = batch

    def project(self, X):
        pass

    def train(self, X, T, delimiter=" "):
        """Trains ELM, can use any X and T(=Y), and specify neurons.

        Neurons: (number, type, [W], [B])
        """

        # get parameters of new data and add neurons
        self.Xmean, self.Xstd = meanstdX(X, self.batch, delimiter)
        if self.classification:
            self.C_dict = c_dictT(T, self.batch)

        # get data iterators
        genX, self.inputs, N = batchX(X, self.batch, delimiter)
        genT, self.targets = batchT(T, self.batch, delimiter, self.C_dict)

        # get mean value of targets
        if self.classification or self.multiclass:
            self.Tmean = np.zeros((self.targets,))  # for any classification
        else:
            self.Tmean, _ = meanstdX(T, self.batch, delimiter)

        # project data
        nn = len(self.ufunc)
        HH = np.zeros((nn, nn))
        HT = np.zeros((nn, self.targets))
        for X, T in zip(genX, genT):

            # get hidden layer outputs
            H = np.dot(X, self.W)
            for i in xrange(H.shape[1]):
                H[:, i] = self.ufunc[i](H[:, i])
            H, T = semi_Tikhonov(H, T, self.Tmean)  # add Tikhonov regularization

            # least squares solution - multiply both sides by H'
            p = float(X.shape[0]) / N
            HH += np.dot(H.T, H)*p
            HT += np.dot(H.T, T)*p

        # solve ELM model
        HH += self.cI * np.eye(nn)  # enhance solution stability
        self.B = lstsq(HH, HT)[0]
        #self.B = np.linalg.pinv(HH).dot(HT)

    def predict(self, X, delimiter=" "):
        """Get predictions using a trained or loaded ELM model.

        :param X: input data
        :rtype: predictions Th
        """

        assert self.B is not None, "train this model first"
        genX, inputs, _ = batchX(X, self.batch, delimiter)

        results = []
        for X in genX:
            assert self.inputs == inputs, "incorrect dimensionality of inputs"
            # project test inputs to outputs
            H = np.dot(X, self.W)
            for i in xrange(H.shape[1]):
                H[:, i] = self.ufunc[i](H[:, i])
            Th1 = H.dot(self.B)
            # additional processing for classification
            if self.classification:
                Th1 = decode(Th1, self.C_dict)
            results.append(Th1)

        # merge results
        if isinstance(results[0], np.ndarray):
            Th = np.vstack(results)
        else:
            Th = []  # merge results which are lists of items
            for r1 in results:
                Th.extend(r1)

        return Th

    def MSE(self, X, Y, delimiter=" "):
        """Mean Squared Error (or mis-classification error).
        """
        MSE = 0
        genX, _, N = batchX(X, self.batch, delimiter)
        genT, _ = batchT(Y, self.batch, delimiter, self.C_dict)

        for X, T in zip(genX, genT):
            H = np.dot(X, self.W)
            for i in xrange(H.shape[1]):
                H[:, i] = self.ufunc[i](H[:, i])
            Th1 = H.dot(self.B)

            p = float(X.shape[0]) / N
            MSE += mse(T, Th1, self.classification, self.multiclass) * p

        return MSE
