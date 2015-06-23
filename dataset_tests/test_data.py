 # -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:59:06 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
import os
import sys

import hpelm

class TestAllDatasets(TestCase):

    def test_SigmClassification_Iris_BetterThanNaive(self):
        X = np.loadtxt("iris/iris_data.txt")
        T = np.loadtxt("iris/iris_classes.txt")
        elm = hpelm.ELM(4, 3)
        elm.add_neurons(10, "sigm")
        elm.train(X, T, 'c')
        Y = elm.predict(X)
        err = elm.error(Y, T)
        self.assertLess(err, 0.66)

    def test_RBFClassification_Iris_BetterThanNaive(self):
        X = np.loadtxt("iris/iris_data.txt")
        T = np.loadtxt("iris/iris_classes.txt")
        elm = hpelm.ELM(4, 3)
        elm.add_neurons(10, "rbf_l2")
        elm.train(X, T, 'c')
        Y = elm.predict(X)
        err = elm.error(Y, T)
        self.assertLess(err, 0.66)

    def test_SigmRegression_Sine_BetterThanNaive(self):
        X = np.loadtxt("sine/sine_x.txt")
        T = np.loadtxt("sine/sine_t.txt")
        elm = hpelm.ELM(1, 1)
        elm.add_neurons(10, "sigm")
        elm.train(X, T)
        Y = elm.predict(X)
        err = elm.error(Y, T)
        self.assertLess(err, 1)

    def test_HPELM_Sine_BetterThanNaive(self):
        X = "sine/sine_x.h5"
        T = "sine/sine_t.h5"
        Y = "sine/sine_y.h5"
        elm = hpelm.HPELM(1, 1)
        elm.add_neurons(10, "sigm")
        elm.train(X, T)
        elm.predict(X, Y)
        err = elm.error(Y, T)
        self.assertLess(err, 1)














