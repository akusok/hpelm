# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:59:06 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
import os

import hpelm


def classification_accuracy(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, 'c')
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def regression_accuracy(folder, nn):
    folder = os.path.join(os.path.dirname(__file__), folder)
    mse = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, "sigm")
        elm.train(Xtr, Ttr)
        Yts = elm.predict(Xts)
        # evaluate classification results
        mse[i - 1] = np.mean((Tts - Yts) ** 2)
    return mse


class TestAllDatasets(TestCase):

    # how much worse our result can be
    # tol = 1.10 means 10% worse
    # tol = 0.90 means 10% better
    tolerance = 1.10

    def test_Sigm_ClassificationBenchmark_Iris(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris", 10, "sigm")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_Sigm_ClassificationBenchmark_Iris_lin(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris", 10, "lin")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_Tanh(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris", 10, "tanh")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_l1(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris", 10, "rbf_l1")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_l2(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris", 10, "rbf_l2")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_linf(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Iris", 10, "rbf_linf")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Pima(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Pima_Indians_Diabetes", 10)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Wine(self):
        target = 81.8 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Wine", 10)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Wisconsin(self):
        target = 95.6 / 100  # from OP-ELM paper
        acc = classification_accuracy("Classification-Wisconsin_Breast_Cancer", 20)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_RegressionBenchmark_Abalone(self):
        target = 8.3  # from OP-ELM paper
        mse = regression_accuracy("Regression-Abalone", 20)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Ailerons(self):
        target = 3.3E-8  # from OP-ELM paper
        mse = regression_accuracy("Regression-Ailerons", 20)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Auto(self):
        target = 7.9E+9  # from OP-ELM paper
        mse = regression_accuracy("Regression-Auto_price", 10)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Bank(self):
        target = 6.7E-3  # from OP-ELM paper
        mse = regression_accuracy("Regression-Bank", 20)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Boston(self):
        target = 1.2E+2  # from OP-ELM paper
        mse = regression_accuracy("Regression-Boston", 10)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Breast_cancer(self):
        target = 7.7E+3  # from OP-ELM paper
        mse = regression_accuracy("Regression-Breast_cancer", 10)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Computer(self):
        target = 4.9E+2  # from OP-ELM paper
        mse = regression_accuracy("Regression-Computer", 10)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_CPU(self):
        target = 4.7E+4  # from OP-ELM paper
        mse = regression_accuracy("Regression-CPU", 10)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Elevators(self):
        target = 2.2E-6  # from OP-ELM paper
        mse = regression_accuracy("Regression-Elevators", 20)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Servo(self):
        target = 7.1  # from OP-ELM paper
        mse = regression_accuracy("Regression-Servo", 10)
        self.assertLess(mse.mean(), target * self.tolerance)

    def test_RegressionBenchmark_Stocks(self):
        target = 3.4E+1  # from OP-ELM paper
        mse = regression_accuracy("Regression-Stocks", 20)
        self.assertLess(mse.mean(), target * self.tolerance)

















