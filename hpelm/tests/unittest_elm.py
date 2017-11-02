# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:04:41 2016

@author: akusok
"""

import os
import tempfile
from unittest import TestCase
import numpy as np
import sys
from hpelm import ELM


class TestCorrectness(TestCase):

    def test_AddNeurons_WorksWithLongType(self):
        if sys.version_info[0] == 2:
            ltype = long
        else:
            ltype = int
        model = ELM(3, 2)
        L = ltype(10)
        model.add_neurons(L, 'tanh')

    def test_CrossValidation_ReturnsError(self):
        model = ELM(5, 2)
        model.add_neurons(10, 'tanh')
        X = np.random.rand(100, 5)
        T = np.random.rand(100, 2)
        err = model.train(X, T, 'CV', k=3)
        self.assertIsNotNone(err)

    def test_ClassificationError_CorrectWithMultipleClasses(self):
        T = np.zeros((100, 5))
        T[:, 0] = 1
        Y = np.zeros((100, 5))
        Y[:, 1] = 1
        model = ELM(1, 5, classification='c')
        self.assertEqual(1, model.error(T, Y))

    def test_MultilabelError_CorrectWithMultipleClasses(self):
        T = np.zeros((100, 5))
        T[:, 0] = 1
        Y = np.zeros((100, 5))
        Y[:, 1] = 1
        model = ELM(1, 5, classification='ml')
        self.assertEqual(0.4, model.error(T, Y))

    def test_LOO_CanSelectMoreThanOneNeuron(self):
        X = np.random.rand(100, 5)
        T = np.random.rand(100, 2)
        for _ in range(10):
            model = ELM(5, 2)
            model.add_neurons(5, 'lin')
            model.train(X, T, 'LOO')
            max1 = model.nnet.L
            if max1 > 1:
                break
        self.assertGreater(max1, 1)

    def test_LOOandOP_CanSelectMoreThanOneNeuron(self):
        X = np.random.rand(100, 5)
        T = np.random.rand(100, 2)
        for _ in range(10):
            model = ELM(5, 2)
            model.add_neurons(5, 'lin')
            model.train(X, T, 'LOO', 'OP')
            max2 = model.nnet.L
            if max2 > 1:
                break
        self.assertGreater(max2, 1)
