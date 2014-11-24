# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:30:05 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
import os


import hpelm


class TestVariableSelection(TestCase):

    tolerance = 1.10

    def test_MRSR_Abalone(self):
        target = 8.3  # from OP-ELM paper

        folder = os.path.join(os.path.dirname(__file__), "../../datasets", "Regression-Abalone")
        mse = np.empty((10,))
        for i in range(1,11):  # 10-fold cross-validation
            # get file names
            Xtr = os.path.join(folder, "xtrain_%d.h5" % i)
            Xts = os.path.join(folder, "xtest_%d.h5" % i)
            Ytr = os.path.join(folder, "ytrain_%d.h5" % i)
            Yts = os.path.join(folder, "ytest_%d.h5" % i)
            # train ELM
            elm = hpelm.ELM()
            elm.train(Xtr, Ytr)
            elm.prune_op(Xtr, Ytr)
            Yh = elm.predict(Xts)

            # evaluate classification results
            Yts = hpelm.h5read(Yts)
            mse[i-1] = np.mean((Yts - Yh)**2)

        self.assertLess(mse.mean(), target*self.tolerance)














#  1. Run ELM on all required datasets

def classification_accuracy(folder):
    folder = os.path.join(os.path.dirname(__file__), "../../datasets", folder)
    acc = np.empty((10,))
    for i in range(1,11):  # 10-fold cross-validation
        # get file names
        Xtr = os.path.join(folder, "xtrain_%d.h5" % i)
        Xts = os.path.join(folder, "xtest_%d.h5" % i)
        Ytr = os.path.join(folder, "ytrain_%d.h5" % i)
        Yts = os.path.join(folder, "ytest_%d.h5" % i)
        # get correct class encoding
        Ytr = hpelm.h5read(Ytr)
        Yts = hpelm.h5read(Yts)
        # train ELM
        elm = hpelm.ELM("classification")
        elm.train(Xtr, Ytr)
        nn = len(elm.ufunc)
        elm.prune_op(Xtr, Ytr)
        print folder[-10:], nn, len(elm.ufunc)
        Yh = elm.predict(Xts)
        # evaluate classification results
        Yh = np.array(Yh).reshape((-1,1))            
        acc[i-1] = float(np.sum(Yh == Yts)) / Yts.shape[0]
    return acc


def regression_accuracy(folder):
    folder = os.path.join(os.path.dirname(__file__), "../../datasets", folder)
    mse = np.empty((10,))
    for i in range(1,11):  # 10-fold cross-validation
        # get file names
        Xtr = os.path.join(folder, "xtrain_%d.h5" % i)
        Xts = os.path.join(folder, "xtest_%d.h5" % i)
        Ytr = os.path.join(folder, "ytrain_%d.h5" % i)
        Yts = os.path.join(folder, "ytest_%d.h5" % i)
        # train ELM
        elm = hpelm.ELM()
        elm.train(Xtr, Ytr)
        elm.prune_op(Xtr, Ytr)
        Yh = elm.predict(Xts)
        # evaluate classification results
        Yts = hpelm.h5read(Yts)
        mse[i-1] = np.mean((Yts - Yh)**2)
    return mse
    

class TestAllDatasets_Pruned(TestCase):
    
    # how much worse our result can be
    # tol = 1.10 means 10% worse
    # tol = 0.90 means 10% better
    tolerance = 1.05
    
    def test_ClassificationBenchmark_Iris(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris")
        self.assertGreater(acc.mean(), target/self.tolerance)
    
    def test_ClassificationBenchmark_Pima(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Pima_Indians_Diabetes")
        self.assertGreater(acc.mean(), target/self.tolerance)

    def test_ClassificationBenchmark_Wine(self):
        target = 81.8 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Wine")
        self.assertGreater(acc.mean(), target/self.tolerance)

    def test_ClassificationBenchmark_Wisconsin(self):
        target = 95.6 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Wisconsin_Breast_Cancer")
        self.assertGreater(acc.mean(), target/self.tolerance)
        

    def test_RegressionBenchmark_Abalone(self):
        target = 8.3  # from OP-ELM paper
        mse = regression_accuracy("Regression-Abalone")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Ailerons(self):
        target = 3.3E-8  # from OP-ELM paper
        mse = regression_accuracy("Regression-Ailerons")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Auto(self):
        target = 7.9E+9  # from OP-ELM paper
        mse = regression_accuracy("Regression-Auto_price")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Bank(self):
        target = 6.7E-3  # from OP-ELM paper
        mse = regression_accuracy("Regression-Bank")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Boston(self):
        target = 1.2E+2  # from OP-ELM paper
        mse = regression_accuracy("Regression-Boston")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Breast_cancer(self):
        target = 7.7E+3  # from OP-ELM paper
        mse = regression_accuracy("Regression-Breast_cancer")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Computer(self):
        target = 4.9E+2  # from OP-ELM paper
        mse = regression_accuracy("Regression-Computer")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_CPU(self):
        target = 4.7E+4  # from OP-ELM paper
        mse = regression_accuracy("Regression-CPU")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Elevators(self):
        target = 2.2E-6  # from OP-ELM paper
        mse = regression_accuracy("Regression-Elevators")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Servo(self):
        target = 7.1  # from OP-ELM paper
        mse = regression_accuracy("Regression-Servo")
        self.assertLess(mse.mean(), target*self.tolerance)

    def test_RegressionBenchmark_Stocks(self):
        target = 3.4E+1  # from OP-ELM paper
        mse = regression_accuracy("Regression-Stocks")
        self.assertLess(mse.mean(), target*self.tolerance)

