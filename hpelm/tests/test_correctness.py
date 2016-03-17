# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

import os
import tempfile
from unittest import TestCase

import numpy as np

from hpelm import ELM


class TestCorrectness(TestCase):
    def test_NonNumpyInputs_RaiseError(self):
        X = np.array([['1', '2'], ['3', '4'], ['5', '6']])
        T = np.array([[1], [2], [3]])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_NonNumpyTargets_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([['a'], ['b'], ['c']])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_OneDimensionInputs_RunsCorrectly(self):
        X = np.array([1, 2, 3])
        T = np.array([[1], [2], [3]])
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_OneDimensionTargets_RunsCorrectly(self):
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_WrongDimensionalityInputs_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2], [3]])
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_WrongDimensionalityTargets_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2], [3]])
        elm = ELM(1, 2)
        elm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_ZeroInputs_RunsCorrectly(self):
        X = np.array([[0, 0], [0, 0], [0, 0]])
        T = np.array([1, 2, 3])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_OneDimensionTargets2_RunsCorrectly(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[0], [0], [0]])
        elm = ELM(2, 1)
        elm.add_neurons(1, "lin")
        elm.train(X, T)

    def test_TrainWithoutNeurons_RaiseError(self):
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm = ELM(1, 1)
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_DifferentNumberOfSamples_RaiseError(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        T = np.array([[1], [2]])
        elm = ELM(2, 1)
        self.assertRaises(AssertionError, elm.train, X, T)

    def test_LinearNeurons_MoreThanInputs_Truncated(self):
        elm = ELM(2, 1)
        elm.add_neurons(3, "lin")
        self.assertEqual(2, elm.nnet.get_neurons()[0][0])

    def test_LinearNeurons_DefaultMatrix_Identity(self):
        elm = ELM(4, 1)
        elm.add_neurons(3, "lin")
        np.testing.assert_array_almost_equal(np.eye(4, 3), elm.nnet.get_neurons()[0][2])

    def test_SLFN_AddLinearNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        self.assertEquals("lin", elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddSigmoidalNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "sigm")
        self.assertEquals("sigm", elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddTanhNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "tanh")
        self.assertEquals("tanh", elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddRbfL1Neurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "rbf_l1")
        self.assertEquals("rbf_l1", elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddRbfL2Neurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "rbf_l2")
        self.assertEquals("rbf_l2", elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddRbfLinfNeurons_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "rbf_linf")
        self.assertEquals("rbf_linf", elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddUfuncNeurons_GotThem(self):
        elm = ELM(1, 1)
        func = np.frompyfunc(lambda a: a + 1, 1, 1)
        elm.add_neurons(1, func)
        self.assertIs(func, elm.nnet.get_neurons()[0][1])

    def test_SLFN_AddTwoNeuronTypes_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.add_neurons(1, "sigm")
        self.assertEquals(2, len(elm.nnet.get_neurons()))
        ntypes = [nr[1] for nr in elm.nnet.get_neurons()]
        self.assertIn("lin", ntypes)
        self.assertIn("sigm", ntypes)

    def test_SLFN_AddNeuronsTwice_GotThem(self):
        elm = ELM(1, 1)
        elm.add_neurons(1, "lin")
        elm.add_neurons(1, "lin")
        self.assertEquals(1, len(elm.nnet.get_neurons()))
        self.assertEquals(2, elm.nnet.get_neurons()[0][0])

    def test_AddNeurons_InitBias_BiasInModel(self):
        elm = ELM(1, 1)
        bias = np.array([1, 2, 3])
        elm.add_neurons(3, "sigm", None, bias)
        np.testing.assert_array_almost_equal(bias, elm.nnet.get_neurons()[0][3])

    def test_AddNeurons_InitW_WInModel(self):
        elm = ELM(2, 1)
        W = np.array([[1, 2, 3], [4, 5, 6]])
        elm.add_neurons(3, "sigm", W, None)
        np.testing.assert_array_almost_equal(W, elm.nnet.get_neurons()[0][2])

    def test_AddNeurons_InitDefault_BiasWNotZero(self):
        elm = ELM(2, 1)
        elm.add_neurons(3, "sigm")
        W = elm.nnet.get_neurons()[0][2]
        bias = elm.nnet.get_neurons()[0][3]
        self.assertGreater(np.sum(np.abs(W)), 0.001)
        self.assertGreater(np.sum(np.abs(bias)), 0.001)

    def test_AddNeurons_InitTwiceBiasW_CorrectlyMerged(self):
        elm = ELM(2, 1)
        W1 = np.random.rand(2, 3)
        W2 = np.random.rand(2, 4)
        bias1 = np.random.rand(3, )
        bias2 = np.random.rand(4, )
        elm.add_neurons(3, "sigm", W1, bias1)
        elm.add_neurons(4, "sigm", W2, bias2)
        np.testing.assert_array_almost_equal(np.hstack((W1, W2)), elm.nnet.get_neurons()[0][2])
        np.testing.assert_array_almost_equal(np.hstack((bias1, bias2)), elm.nnet.get_neurons()[0][3])

    def test_Str_Works(self):
        elm = ELM(1, 1)
        s = "%s" % elm
        self.assertIn("ELM with 1 inputs and 1 output", s)

    def test_StrCustomNeurons_DisplaysName(self):
        elm = ELM(1, 1)
        func = np.sin
        elm.add_neurons(1, func)
        s_elm = "%s" % elm
        self.assertIn("sin", s_elm)

    def test_ELMWithBatch_SetsBatch(self):
        elm = ELM(1, 1, batch=123)
        self.assertEqual(123, elm.batch)

    def test_TrainWithBatch_OverwritesBatch(self):
        elm = ELM(1, 1, batch=123)
        X = np.array([1, 2, 3])
        T = np.array([1, 2, 3])
        elm.add_neurons(1, "lin")
        elm.train(X, T, batch=234)
        self.assertEqual(234, elm.batch)

    def test_Classification_Works(self):
        elm = ELM(1, 2)
        X = np.array([1, 2, 3, 4, 5, 6])
        T = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        elm.add_neurons(1, "lin")
        elm.train(X, T, 'c')

    def test_Classification_WorksCorreclty(self):
        elm = ELM(1, 2)
        X = np.array([-1, -0.6, -0.3, 0.3, 0.6, 1])
        T = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        elm.add_neurons(1, "lin")
        elm.train(X, T, 'c')
        Y = elm.predict(X)
        self.assertGreater(Y[0, 0], Y[0, 1])
        self.assertLess(Y[5, 0], Y[5, 1])

    def test_MultiLabelClassification_Works(self):
        elm = ELM(1, 2)
        X = np.array([1, 2, 3, 4, 5, 6])
        T = np.array([[1, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])
        elm.add_neurons(1, "lin")
        elm.train(X, T, 'ml')
        elm.train(X, T, 'mc')

    def test_WeightedClassification_Works(self):
        elm = ELM(1, 2)
        X = np.array([1, 2, 3, 1, 2, 3])
        T = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        elm.add_neurons(1, "lin")
        elm.train(X, T, 'wc', w=(1, 1))

    def test_WeightedClassification_DefaultWeightsWork(self):
        elm = ELM(1, 2)
        X = np.array([1, 2, 3, 1, 2, 3])
        T = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        elm.add_neurons(1, "lin")
        elm.train(X, T, 'wc')

    def test_WeightedClassification_ClassWithLargerWeightWins(self):
        elm = ELM(1, 2)
        X = np.array([1, 2, 3, 1, 2, 3])
        T = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        elm.add_neurons(1, "lin")
        elm.train(X, T, 'wc', w=(1, 0.1))
        Y = elm.predict(X)
        self.assertGreater(Y[0, 0], Y[0, 1])
        self.assertGreater(Y[1, 0], Y[1, 1])
        self.assertGreater(Y[2, 0], Y[2, 1])

    def test_RegressionError_Works(self):
        T = np.array([1, 2, 3])
        Y = np.array([1.1, 2.2, 3.3])
        err1 = np.mean((T - Y) ** 2)
        elm = ELM(1, 1)
        e = elm.error(T, Y)
        np.testing.assert_allclose(e, err1)

    def test_ClassificationError_Works(self):
        X = np.array([1, 2, 3])
        T = np.array([[0, 1], [0, 1], [1, 0]])
        Y = np.array([[0, 1], [0.4, 0.6], [0, 1]])
        elm = ELM(1, 2, classification='c')
        elm.add_neurons(1, "lin")
        e = elm.error(T, Y)
        np.testing.assert_allclose(e, 1.0 / 3)

    def test_WeightedClassError_Works(self):
        T = np.array([[0, 1], [0, 1], [1, 0]])
        Y = np.array([[0, 1], [0.4, 0.6], [0, 1]])
        # here class 0 is totally incorrect, and class 1 is totally correct
        w = (9, 1)
        elm = ELM(1, 2, classification="wc", w=w)
        elm.add_neurons(1, "lin")
        e = elm.error(T, Y)
        np.testing.assert_allclose(e, 0.9)

    def test_MultiLabelClassError_Works(self):
        X = np.array([1, 2, 3])
        T = np.array([[0, 1], [1, 1], [1, 0]])
        Y = np.array([[0.4, 0.6], [0.8, 0.6], [1, 1]])
        elm = ELM(1, 2, classification="ml")
        elm.add_neurons(1, "lin")
        e = elm.error(T, Y)
        np.testing.assert_allclose(e, 1.0 / 6)

    def test_ProjectELM_WorksCorrectly(self):
        X = np.array([[1], [2], [3]])
        elm = ELM(1, 1)
        elm.add_neurons(1, "tanh", np.array([[1]]), np.array([0]))
        H = elm.project(X)
        np.testing.assert_allclose(H, np.tanh(X))

    def test_InitELM_SetNorm(self):
        nr = 0.03
        elm = ELM(1, 1, norm=nr)
        self.assertEqual(nr, elm.nnet.norm)

    def test_PrecisionELM_UsesPrecision(self):
        elm1 = ELM(1, 1, precision='32')
        self.assertIs(elm1.nnet.precision, np.float32)
        elm2 = ELM(1, 1, precision='single')
        self.assertIs(elm2.nnet.precision, np.float32)
        elm3 = ELM(1, 1, precision=np.float32)
        self.assertIs(elm3.nnet.precision, np.float32)
        elm4 = ELM(1, 1, precision='64')
        self.assertIs(elm4.nnet.precision, np.float64)
        elm5 = ELM(1, 1, precision='double')
        self.assertIs(elm5.nnet.precision, np.float64)
        elm6 = ELM(1, 1, precision=np.float64)
        self.assertIs(elm6.nnet.precision, np.float64)
        elm7 = ELM(1, 1)  # default double precision
        self.assertIs(elm7.nnet.precision, np.float64)
        elm8 = ELM(1, 1, precision="lol")  # default double precision
        self.assertIs(elm8.nnet.precision, np.float64)

    def test_ELM_SaveLoad(self):
        X = np.array([1, 2, 3, 1, 2, 3])
        T = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        elm = ELM(1, 2, precision='32', norm=0.02)
        elm.add_neurons(1, "lin")
        elm.add_neurons(2, "tanh")
        elm.train(X, T, "wc", w=(0.7, 0.3))
        B1 = elm.nnet.get_B()
        try:
            f, fname = tempfile.mkstemp()
            elm.save(fname)
            elm2 = ELM(3, 3)
            elm2.load(fname)
        finally:
            os.close(f)
        self.assertEqual(elm2.nnet.inputs, 1)
        self.assertEqual(elm2.nnet.outputs, 2)
        self.assertEqual(elm2.classification, "wc")
        self.assertIs(elm.precision, np.float32)
        self.assertIs(elm2.precision, np.float64)  # precision has changed
        np.testing.assert_allclose(np.array([0.7, 0.3]), elm2.wc)
        np.testing.assert_allclose(0.02, elm2.nnet.norm)
        np.testing.assert_allclose(B1, elm2.nnet.get_B())
        self.assertEqual(elm2.nnet.get_neurons()[0][1], "lin")
        self.assertEqual(elm2.nnet.get_neurons()[1][1], "tanh")

    def test_SaveELM_WrongFile(self):
        elm = ELM(1, 1)
        try:
            f, fname = tempfile.mkstemp()
            self.assertRaises(IOError, elm.save, os.path.dirname(fname) + "olo/lo")
        finally:
            os.close(f)

    def test_LoadELM_WrongFile(self):
        elm = ELM(1, 1)
        try:
            f, fname = tempfile.mkstemp()
            self.assertRaises(IOError, elm.load, fname + "ololo2")
        finally:
            os.close(f)

    def test_ConfusionELM_Classification(self):
        T = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        Y = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])
        elm = ELM(1, 2)
        elm.classification = "c"
        C = elm.confusion(T, Y)
        np.testing.assert_allclose(C, np.array([[0, 2], [1, 1]]))

    def test_ConfusionELM_Multilabel(self):
        T = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        Y = np.array([[1, 1], [1, 0], [0, 1], [0, 1]])
        elm = ELM(1, 2)
        elm.classification = "ml"
        C = elm.confusion(T, Y)
        np.testing.assert_allclose(C, np.array([[2, 1], [0, 2]]))

    def test_MRSR_Works(self):
        X = np.random.rand(10, 3)
        T = np.random.rand(10, 2)
        elm = ELM(3, 2)
        elm.add_neurons(5, "tanh")
        elm.train(X, T, "LOO", "OP")

    def test_MRSR2_Works(self):
        X = np.random.rand(20, 9)
        T = np.random.rand(20, 12)
        elm = ELM(9, 12)
        elm.add_neurons(5, "tanh")
        elm.train(X, T, "LOO", "OP")





