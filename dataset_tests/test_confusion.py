# -*- coding: utf-8 -*-
"""Check that confusion matrix bug will not repeat.

Created on Sun Nov  1 23:17:48 2015

@author: akusok
"""


from unittest import TestCase
import numpy as np
import os

import hpelm


def classification_conf(folder, nn, ntype="sigm", b=1):
    folder = os.path.join(os.path.dirname(__file__), folder)
    i = np.random.randint(0, 10)
    print "using init number: ", i
    # get file names
    Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
    Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
    Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
    Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
    # train ELM
    Bsize = Xtr.shape[0]/b + 1  # batch size larger than amount of data
    elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1], batch = Bsize)
    elm.add_neurons(nn, ntype)
    elm.train(Xtr, Ttr, 'c')
    Yts = elm.predict(Xts)
    conf = elm.confusion(Tts, Yts)
    return conf



class TestAllDatasets(TestCase):

    # how much worse our result can be
    # tol = 1.10 means 10% worse
    # tol = 0.90 means 10% better
    tolerance = 1.10

    def test_ConfusionSingleBatch_Iris_NonZero(self):
        conf = classification_conf("Classification-Iris", 10, "sigm", b=1)
        self.assertGreater(conf.sum(), 0)

    def test_ConfusionMultiBatch_Iris_NonZero(self):
        conf = classification_conf("Classification-Iris", 10, "sigm", b=10)
        self.assertGreater(conf.sum(), 0)

    def test_ConfusionUnitBatch_Iris_NonZero(self):
        conf = classification_conf("Classification-Iris", 10, "sigm", b=100000)
        self.assertGreater(conf.sum(), 0)

    def test_ConfusionSingleBatch_Pima_NonZero(self):
        conf = classification_conf("Classification-Pima_Indians_Diabetes", 10, "sigm", b=1)
        self.assertGreater(conf.sum(), 0)

    def test_ConfusionSingleBatch_Wine_NonZero(self):
        conf = classification_conf("Classification-Wine", 10, "sigm", b=1)
        self.assertGreater(conf.sum(), 0)

    def test_ConfusionSingleBatch_Wisconsin_NonZero(self):
        conf = classification_conf("Classification-Wisconsin_Breast_Cancer", 10, "sigm", b=1)
        self.assertGreater(conf.sum(), 0)














