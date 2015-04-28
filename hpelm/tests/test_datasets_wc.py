# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:05:07 2015

@author: akusok
"""


from unittest import TestCase
import numpy as np
import os

import hpelm


def accuracy_wc_loo(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), "../../datasets", folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        w = np.ones((Ttr.shape[1],))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "wc", "LOO", w=w)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def accuracy_wc_cv(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), "../../datasets", folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        w = np.ones((Ttr.shape[1],))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "wc", "CV", k=5, w=w)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


class TestMulticlassLOO(TestCase):
    # how much worse our result can be
    # tol = 1.10 means 10% worse
    # tol = 0.90 means 10% better
    tolerance = 1.05

    def test_Sigm_ClassificationBenchmark_Iris(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Iris", 15, "sigm")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_Sigm_ClassificationBenchmark_Iris_lin(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Iris", 15, "lin")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_Tanh(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Iris", 15, "tanh")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_l1(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Iris", 15, "rbf_l1")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_l2(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Iris", 15, "rbf_l2")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_linf(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Iris", 15, "rbf_linf")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Pima(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Pima_Indians_Diabetes", 10)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Wine(self):
        target = 81.8 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Wine", 10)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Wisconsin(self):
        target = 95.6 / 100  # from OP-ELM paper
        acc = accuracy_wc_loo("Classification-Wisconsin_Breast_Cancer", 20)
        self.assertGreater(acc.mean(), target / self.tolerance)


class TestMulticlassCV(TestCase):
    # how much worse our result can be
    # tol = 1.10 means 10% worse
    # tol = 0.90 means 10% better
    tolerance = 1.05

    def test_Sigm_ClassificationBenchmark_Iris(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Iris", 15, "sigm")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_Sigm_ClassificationBenchmark_Iris_lin(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Iris", 15, "lin")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_Tanh(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Iris", 15, "tanh")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_l1(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Iris", 15, "rbf_l1")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_l2(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Iris", 15, "rbf_l2")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Iris_RBF_linf(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Iris", 15, "rbf_linf")
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Pima(self):
        target = 72.2 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Pima_Indians_Diabetes", 10)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Wine(self):
        target = 81.8 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Wine", 10)
        self.assertGreater(acc.mean(), target / self.tolerance)

    def test_ClassificationBenchmark_Wisconsin(self):
        target = 95.6 / 100  # from OP-ELM paper
        acc = accuracy_wc_cv("Classification-Wisconsin_Breast_Cancer", 20)
        self.assertGreater(acc.mean(), target / self.tolerance)














