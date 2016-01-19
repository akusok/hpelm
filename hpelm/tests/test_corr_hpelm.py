# -*- coding: utf-8 -*-
"""Copy of test_correctness.py
Created on Wed Sep 23 21:15:18 2015

@author: akusok
"""


from unittest import TestCase
import numpy as np
import tempfile
import os

from hpelm import HPELM
from hpelm.modules.hdf5_tools import make_hdf5


# noinspection PyArgumentList
class TestCorrectness(TestCase):
    tfiles = None

    def makeh5(self, data):
        f, fname = tempfile.mkstemp()
        os.close(f)
        self.tfiles.append(fname)
        make_hdf5(data, fname)
        return fname

    def makefile(self):
        f, fname = tempfile.mkstemp()
        os.close(f)
        os.remove(fname)
        self.tfiles.append(fname)
        return fname

    def setUp(self):
        self.tfiles = []

    def tearDown(self):
        for fname in self.tfiles:
            os.remove(fname)

    def test_NonNumpyInputs_RaiseError(self):
        X = np.array([['1', '2'], ['3', '4'], ['5', '6']])
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, hpelm.train, X, T)

    def test_NonNumpyTargets_RaiseError(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = np.array([['a'], ['b'], ['c']])
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, hpelm.train, X, T)

    def test_OneDimensionInputs_RunsCorrectly(self):
        X = self.makeh5(np.array([1, 2, 3]))
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)

    def test_OneDimensionTargets_RunsCorrectly(self):
        X = self.makeh5(np.array([1, 2, 3]))
        T = self.makeh5(np.array([1, 2, 3]))
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)

    def test_WrongDimensionalityInputs_RaiseError(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, hpelm.train, X, T)

    def test_WrongDimensionalityTargets_RaiseError(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        self.assertRaises(AssertionError, hpelm.train, X, T)

    def test_ZeroInputs_RunsCorrectly(self):
        X = self.makeh5(np.array([[0, 0], [0, 0], [0, 0]]))
        T = self.makeh5(np.array([1, 2, 3]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)

    def test_OneDimensionTargets2_RunsCorrectly(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[0], [0], [0]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)

    def test_TrainWithoutNeurons_RaiseError(self):
        X = self.makeh5(np.array([1, 2, 3]))
        T = self.makeh5(np.array([1, 2, 3]))
        hpelm = HPELM(1, 1)
        self.assertRaises(AssertionError, hpelm.train, X, T)

    def test_DifferentNumberOfSamples_RaiseError(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[1], [2]]))
        hpelm = HPELM(2, 1)
        self.assertRaises(AssertionError, hpelm.train, X, T)

    def test_LinearNeurons_MoreThanInputs_Truncated(self):
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        self.assertEqual(2, hpelm.nnet.get_neurons()[0][0])

    def test_LinearNeurons_DefaultMatrix_Identity(self):
        hpelm = HPELM(4, 1)
        hpelm.add_neurons(3, "lin")
        np.testing.assert_array_almost_equal(np.eye(4, 3), hpelm.nnet.get_neurons()[0][2])

    def test_SLFN_AddLinearNeurons_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "lin")
        self.assertEquals("lin", hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddSigmoidalNeurons_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "sigm")
        self.assertEquals("sigm", hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddTanhNeurons_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "tanh")
        self.assertEquals("tanh", hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddRbfL1Neurons_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "rbf_l1")
        self.assertEquals("rbf_l1", hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddRbfL2Neurons_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "rbf_l2")
        self.assertEquals("rbf_l2", hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddRbfLinfNeurons_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "rbf_linf")
        self.assertEquals("rbf_linf", hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddUfuncNeurons_GotThem(self):
        hpelm = HPELM(1, 1)
        func = np.frompyfunc(lambda a: a+1, 1, 1)
        hpelm.add_neurons(1, func)
        self.assertIs(func, hpelm.nnet.get_neurons()[0][1])

    def test_SLFN_AddTwoNeuronTypes_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.add_neurons(1, "sigm")
        self.assertEquals(2, len(hpelm.nnet.get_neurons()))
        ntypes = [nr[1] for nr in hpelm.nnet.get_neurons()]
        self.assertIn("lin", ntypes)
        self.assertIn("sigm", ntypes)

    def test_SLFN_AddNeuronsTwice_GotThem(self):
        hpelm = HPELM(1, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.add_neurons(1, "lin")
        self.assertEquals(1, len(hpelm.nnet.get_neurons()))
        self.assertEquals(2, hpelm.nnet.get_neurons()[0][0])

    def test_AddNeurons_InitBias_BiasInModel(self):
        hpelm = HPELM(1, 1)
        bias = np.array([1, 2, 3])
        hpelm.add_neurons(3, "sigm", None, bias)
        neurons = hpelm.nnet.get_neurons()
        np.testing.assert_array_almost_equal(bias, neurons[0][3])

    def test_AddNeurons_InitW_WInModel(self):
        hpelm = HPELM(2, 1)
        W = np.array([[1, 2, 3], [4, 5, 6]])
        hpelm.add_neurons(3, "sigm", W, None)
        np.testing.assert_array_almost_equal(W, hpelm.nnet.get_neurons()[0][2])

    def test_AddNeurons_InitDefault_BiasWNotZero(self):
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "sigm")
        W = hpelm.nnet.get_neurons()[0][2]
        bias = hpelm.nnet.get_neurons()[0][3]
        self.assertGreater(np.sum(np.abs(W)), 0.001)
        self.assertGreater(np.sum(np.abs(bias)), 0.001)

    def test_AddNeurons_InitTwiceBiasW_CorrectlyMerged(self):
        hpelm = HPELM(2, 1)
        W1 = np.random.rand(2, 3)
        W2 = np.random.rand(2, 4)
        bias1 = np.random.rand(3,)
        bias2 = np.random.rand(4,)
        hpelm.add_neurons(3, "sigm", W1, bias1)
        hpelm.add_neurons(4, "sigm", W2, bias2)
        np.testing.assert_array_almost_equal(np.hstack((W1, W2)), hpelm.nnet.get_neurons()[0][2])
        np.testing.assert_array_almost_equal(np.hstack((bias1, bias2)), hpelm.nnet.get_neurons()[0][3])

    def test_TrainIstart_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T, istart=1)

    def test_TrainIcount_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T, icount=2)

    def test_TrainIstart_HasEffect(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[3], [2], [3]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)
        B1 = hpelm.nnet.get_B()
        hpelm.train(X, T, istart=1)
        B2 = hpelm.nnet.get_B()
        self.assertFalse(np.allclose(B1, B2), "iStart index does not work")

    def test_TrainIcount_HasEffect(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[3], [2], [3]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)
        B1 = hpelm.nnet.get_B()
        hpelm.train(X, T, icount=2)
        B2 = hpelm.nnet.get_B()
        self.assertFalse(np.allclose(B1, B2), "iCount index does not work")

    def test_TrainAsync_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6]]))
        T = self.makeh5(np.array([[1], [2], [3]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train_async(X, T)

    def test_TrainAsyncWeighted_Works(self):
        X = self.makeh5(np.array([1, 2, 3, 1, 2, 3]))
        T = self.makeh5(np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]))
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        hpelm.train_async(X, T, 'wc', wc=(1,2))

    def test_TrainAsyncIndexed_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train_async(X, T, istart=1, icount=2)

    def test_WeightedClassification_Works(self):
        X = self.makeh5(np.array([1, 2, 3, 1, 2, 3]))
        T = self.makeh5(np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]))
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T, 'wc', w=(1, 1))

    def test_WeightedClassification_DefaultWeightsWork(self):
        X = self.makeh5(np.array([1, 2, 3, 1, 2, 3]))
        T = self.makeh5(np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]))
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T, 'wc')

    def test_HPELM_tprint(self):
        X = self.makeh5(np.array([1, 2, 3, 1, 2, 3]))
        T = self.makeh5(np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]))
        hpelm = HPELM(1, 2, batch=2, tprint=0)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)

    def test_AddDataToFile_SingleAddition(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)

    def test_AddDataToFile_MultipleAdditions(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)

    def test_AddDataAsyncToFile_SingleAddition(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data_async(X, T, fHH=fHH, fHT=fHT)

    def test_AddDataAsyncToFile_MultipleAdditions(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data_async(X, T, fHH=fHH, fHT=fHT)
        hpelm.add_data_async(X, T, fHH=fHH, fHT=fHT)

    def test_AddDataToFile_MixedSequentialAsync(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)
        hpelm.add_data_async(X, T, fHH=fHH, fHT=fHT)

    def test_SolveCorr_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(3, "lin")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)
        hpelm.solve_corr(fHH, fHT)
        self.assertIsNot(hpelm.nnet.get_B(), None)

    def test_ValidationCorr_Works(self):
        X = self.makeh5(np.random.rand(30, 3))
        T = self.makeh5(np.random.rand(30, 2))
        hpelm = HPELM(3, 2, norm=1e-6)
        hpelm.add_neurons(6, "tanh")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)
        nns, err, confs = hpelm.validation_corr(fHH, fHT, X, T, steps=3)
        self.assertGreater(err[0], err[-1])

    def test_ValidationCorr_ReturnsConfusion(self):
        X = self.makeh5(np.random.rand(10, 3))
        T = self.makeh5(np.random.rand(10, 2))
        hpelm = HPELM(3, 2, classification="c")
        hpelm.add_neurons(6, "tanh")
        fHH = self.makefile()
        fHT = self.makefile()
        hpelm.add_data(X, T, fHH=fHH, fHT=fHT)
        _, _, confs = hpelm.validation_corr(fHH, fHT, X, T, steps=3)
        self.assertGreater(np.sum(confs[0]), 1)

    def test_Predict_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)
        fY = self.makefile()
        hpelm.predict(X, fY)

    def test_PredictAsync_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)
        fY = self.makefile()
        hpelm.predict_async(X, fY)

    def test_Project_Works(self):
        X = self.makeh5(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        T = self.makeh5(np.array([[1], [2], [3], [4]]))
        hpelm = HPELM(2, 1)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T)
        fH = self.makefile()
        hpelm.project(X, fH)

    def test_RegressionError_Works(self):
        T = np.array([1, 2, 3])
        Y = np.array([1.1, 2.2, 3.3])
        err1 = np.mean((T - Y) ** 2)
        fT = self.makeh5(T)
        fY = self.makeh5(Y)
        hpelm = HPELM(1, 1)
        e = hpelm.error(fT, fY)
        np.testing.assert_allclose(e, err1)

    def test_ClassificationError_Works(self):
        T = self.makeh5(np.array([[0, 1], [0, 1], [1, 0]]))
        Y = self.makeh5(np.array([[0, 1], [0.4, 0.6], [0, 1]]))
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        hpelm.classification = "c"
        e = hpelm.error(T, Y)
        np.testing.assert_allclose(e, 1.0 / 3)

    def test_WeightedClassError_Works(self):
        X = self.makeh5(np.array([1, 2, 3]))
        T = self.makeh5(np.array([[0, 1], [0, 1], [1, 0]]))
        Y = self.makeh5(np.array([[0, 1], [0.4, 0.6], [0, 1]]))
        # here class 0 is totally incorrect, and class 1 is totally correct
        w = (9, 1)
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        hpelm.train(X, T, "wc", w=w)
        e = hpelm.error(T, Y)
        np.testing.assert_allclose(e, 0.9)

    def test_MultiLabelClassError_Works(self):
        T = self.makeh5(np.array([[0, 1], [1, 1], [1, 0]]))
        Y = self.makeh5(np.array([[0.4, 0.6], [0.8, 0.6], [1, 1]]))
        hpelm = HPELM(1, 2)
        hpelm.add_neurons(1, "lin")
        hpelm.classification = "ml"
        e = hpelm.error(T, Y)
        np.testing.assert_allclose(e, 1.0 / 6)






