# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np

import hpelm


class TestCorrectness(TestCase):

    def test_1_NonNumpyInputs_RaiseError(self):
        X = [[1, 2], [3, 4], [5, 6]]
        T = np.array([[1], [2], [3]])
        elm = hpelm.ELM(2, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_2_NonNumpyTargets_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = [[1], [2], [3]]
        elm = hpelm.ELM(2, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_3_OneDimensionInputs_RunsCorrectly(self):
        X = np.array([1, 2, 3])
        T = np.array([[1], [2], [3]])
        elm = hpelm.ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_4_OneDimensionTargets_RunsCorrectly(self):
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm = hpelm.ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_5_WrongDimensionalityInputs_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2], [3]])
        elm = hpelm.ELM(1, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_6_WrongDimensionalityTargets_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2], [3]])
        elm = hpelm.ELM(1, 2)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_7_ZeroInputs_RunsCorrectly(self):
        X = np.array([[0, 0], [0, 0], [0, 0]])
        T = np.array([1, 2, 3])
        elm = hpelm.ELM(2, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_8_OneDimensionTargets_RunsCorrectly(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[0], [0], [0]])
        elm = hpelm.ELM(2, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_9_TrainWithoutNeurons_RaiseError(self):
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm = hpelm.ELM(1, 1)
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_10_DifferentNumberOfSamples_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2]])
        elm = hpelm.ELM(2, 1)
        self.assertRaises(AssertionError, elm.train, X, T)
