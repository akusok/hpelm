# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np

from hpelm import ELM


class TestCorrectness(TestCase):

    def test_1_NonNumpyInputs_RaiseError(self):
        X = np.array([['1', '2'], ['3', '4'], ['5', '6']])
        T = np.array([[1], [2], [3]])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_2_NonNumpyTargets_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([['a'], ['b'], ['c']])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_3_OneDimensionInputs_RunsCorrectly(self):
        X = np.array([1, 2, 3])
        T = np.array([[1], [2], [3]])
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_4_OneDimensionTargets_RunsCorrectly(self):
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_5_WrongDimensionalityInputs_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2], [3]])
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_6_WrongDimensionalityTargets_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2], [3]])
        elm = ELM(1, 2)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_7_ZeroInputs_RunsCorrectly(self):
        X = np.array([[0, 0], [0, 0], [0, 0]])
        T = np.array([1, 2, 3])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_8_OneDimensionTargets_RunsCorrectly(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[0], [0], [0]])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_9_TrainWithoutNeurons_RaiseError(self):
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm = ELM(1, 1)
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_10_DifferentNumberOfSamples_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2]])
        elm = ELM(2, 1)
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_11_LinearNeurons_MoreThanInputs_Truncated(self):
        elm = ELM(2, 1)
        elm.add_neurons(3, "lin")
        self.assertEqual(2, elm.neurons[0][1])

    def test_12_LinearNeurons_DefaultMatrix_Identity(self):
        elm = ELM(4, 1)
        elm.add_neurons(3, "lin")
        np.testing.assert_array_almost_equal(np.eye(4, 3), elm.neurons[0][2])

    def test_13_SLFN_AddLinearNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        self.assertEquals("lin", elm.neurons[0][0])

    def test_14_SLFN_AddSigmoidalNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "sigm")
        self.assertEquals("sigm", elm.neurons[0][0])

    def test_15_SLFN_AddTanhNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "tanh")
        self.assertEquals("tanh", elm.neurons[0][0])

    def test_16_SLFN_AddRbfL1Neurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "rbf_l1")
        self.assertEquals("rbf_l1", elm.neurons[0][0])

    def test_17_SLFN_AddRbfL2Neurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "rbf_l2")
        self.assertEquals("rbf_l2", elm.neurons[0][0])

    def test_18_SLFN_AddRbfLinfNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "rbf_linf")
        self.assertEquals("rbf_linf", elm.neurons[0][0])

    def test_19_SLFN_AddUfuncNeurons_GotThem(self):
        elm = ELM(1, 1)
        func = np.frompyfunc(lambda a: a+1, 1, 1)
        elm.add_neurons(1, func)
        self.assertIs(func, elm.neurons[0][0])

    def test_20_SLFN_AddTwoNeuronTypes_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.add_neurons(1, "sigm")
        self.assertEquals(2, len(elm.neurons))
        ntypes = [nr[0] for nr in elm.neurons]
        self.assertIn("lin", ntypes)
        self.assertIn("sigm", ntypes)

    def test_21_SLFN_AddNeuronsTwice_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.add_neurons(1, "lin")
        self.assertEquals(1, len(elm.neurons))
        self.assertEquals(2, elm.neurons[0][1])

    def test_22_AddNeurons_InitBias_BiasInModel(self):
        elm = ELM(1, 1)
        bias = np.array([1, 2, 3])
        elm.add_neurons(3, "sigm", None, bias)
        np.testing.assert_array_almost_equal(bias, elm.neurons[0][3])

    def test_23_AddNeurons_InitW_WInModel(self):
        elm = ELM(2, 1)
        W = np.array([[1, 2, 3], [4, 5, 6]])
        elm.add_neurons(3, "sigm", W, None)
        np.testing.assert_array_almost_equal(W, elm.neurons[0][2])

    def test_24_AddNeurons_InitDefault_BiasWNotZero(self):
        elm = ELM(2, 1)
        elm.add_neurons(3, "sigm")
        W = elm.neurons[0][2]
        bias = elm.neurons[0][3]
        self.assertGreater(np.sum(np.abs(W)), 0.001)
        self.assertGreater(np.sum(np.abs(bias)), 0.001)

    def test_25_AddNeurons_InitTwiceBiasW_CorrectlyMerged(self):
        elm = ELM(2, 1)
        W1 = np.random.rand(2, 3)
        W2 = np.random.rand(2, 4)
        bias1 = np.random.rand(3,)
        bias2 = np.random.rand(4,)
        elm.add_neurons(3, "sigm", W1, bias1)
        elm.add_neurons(4, "sigm", W2, bias2)
        np.testing.assert_array_almost_equal(np.hstack((W1, W2)), elm.neurons[0][2])
        np.testing.assert_array_almost_equal(np.hstack((bias1, bias2)), elm.neurons[0][3])

























