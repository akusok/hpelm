# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:06:28 2016

@author: akusok
"""

import os
import tempfile
from unittest import TestCase
import numpy as np

from hpelm import HPELM, modules


class TestHPELM(TestCase):
    def setUp(self):
        tfile, self.fname = tempfile.mkstemp()
        os.close(tfile)

    def tearDown(self):
        os.remove(self.fname)

    def test_SaveEmptyModel_CanLoad(self):
        model = HPELM(10, 3)
        model.save(self.fname)
        model2 = HPELM(10, 3)
        model2.load(self.fname)

    def test_ClassificationError_CorrectWithMultipleClasses(self):
        T = np.zeros((100, 5))
        T[:, 0] = 1
        Y = np.zeros((100, 5))
        Y[:, 1] = 1
        model = HPELM(1, 5, classification='c')
        self.assertEqual(1, model.error(T, Y))

    def test_MultilabelError_CorrectWithMultipleClasses(self):
        T = np.zeros((100, 5))
        T[:, 0] = 1
        Y = np.zeros((100, 5))
        Y[:, 1] = 1
        model = HPELM(1, 5, classification='ml')
        self.assertEqual(0.4, model.error(T, Y))


class TestParallelELM(TestCase):
    def setUp(self):
        tfile, self.fname = tempfile.mkstemp()
        tmodel, self.fmodel = tempfile.mkstemp()
        tfileX, self.fnameX = tempfile.mkstemp()
        tfileT, self.fnameT = tempfile.mkstemp()
        tfileY, self.fnameY = tempfile.mkstemp()
        tfileHT, self.fnameHT = tempfile.mkstemp()
        tfileHH, self.fnameHH = tempfile.mkstemp()
        os.close(tfile)
        os.close(tmodel)
        os.close(tfileX)
        os.close(tfileT)
        os.close(tfileY)
        os.close(tfileHT)
        os.close(tfileHH)

    def tearDown(self):
        os.remove(self.fname)
        os.remove(self.fmodel)
        os.remove(self.fnameX)
        os.remove(self.fnameT)
        os.remove(self.fnameY)
        os.remove(self.fnameHT)
        os.remove(self.fnameHH)

    def test_ParallelBasicPython_Works(self):
        X = np.random.rand(1000, 10)
        T = np.random.rand(1000, 3)
        hX = modules.make_hdf5(X, self.fnameX)
        hT = modules.make_hdf5(T, self.fnameT)

        model0 = HPELM(10, 3)
        model0.add_neurons(10, 'lin')
        model0.add_neurons(5, 'tanh')
        model0.add_neurons(15, 'sigm')
        model0.save(self.fmodel)

        model1 = HPELM(10, 3)
        model1.load(self.fmodel)
        os.remove(self.fnameHT)
        os.remove(self.fnameHH)
        model1.add_data(self.fnameX, self.fnameT, istart=0, icount=100, fHH=self.fnameHH, fHT=self.fnameHT)

        model2 = HPELM(10, 3)
        model2.load(self.fmodel)
        model2.add_data(self.fnameX, self.fnameT, istart=100, icount=900, fHH=self.fnameHH, fHT=self.fnameHT)

        model3 = HPELM(10, 3)
        model3.load(self.fmodel)
        model3.solve_corr(self.fnameHH, self.fnameHT)
        model3.save(self.fmodel)

        model4 = HPELM(10, 3)
        model4.load(self.fmodel)
        model4.predict(self.fnameX, self.fnameY)

        err = model4.error(self.fnameT, self.fnameY, istart=0, icount=198)
        self.assertLess(err, 1)

        err = model4.error(self.fnameT, self.fnameY, istart=379, icount=872)
        self.assertLess(err, 1)




























