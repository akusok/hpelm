# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
import os

import hpelm


class TestTrainFromFile(TestCase):
    
    def test_Load_TextFiles(self):
        d = os.path.join(os.path.dirname(__file__), "../datasets/sine")
        x = os.path.join(d, "sine_x.txt")        
        y = os.path.join(d, "sine_y.txt")        
        elm = hpelm.ELM()
        elm.train(x,y)
        self.assertIsNotNone(elm.B)

    def test_Load_Hdf5Files(self):
        d = os.path.join(os.path.dirname(__file__), "../datasets/sine")
        x = os.path.join(d, "sine_x.h5")        
        y = os.path.join(d, "sine_y.h5")        
        elm = hpelm.ELM()
        elm.train(x,y)
        self.assertIsNotNone(elm.B)
        
    def test_Load_MatrixAndFileMixedInputs(self):
        d = os.path.join(os.path.dirname(__file__), "../datasets/sine")
        x = os.path.join(d, "sine_x.h5")        
        y = os.path.join(d, "sine_y.txt")        
        y = np.loadtxt(y)
        elm = hpelm.ELM()
        elm.train(x,y)
        self.assertIsNotNone(elm.B)


class TestErrors(TestCase):
    
    def test_LooPress_ReturnsMSE(self):
        x = np.linspace(-1,1,100)
        y = np.sin(x)
        elm = hpelm.ELM()
        elm.train(x,y)
        E = elm.loo_press(x,y)
        assert E < 0.1


class TestDataLoader(TestCase):
    
    def test_InputX_ReshapeAddBias(self):
        x1 = [1,2,3]
        x2 = np.array([[1,1],[2,1],[3,1]])
        elm = hpelm.ELM()
        elm.xmean = 0
        elm.xstd = 1
        x1p = elm.data_loader(x1)
        assert x2.shape == x1p.shape
        assert np.allclose(x2,x1p)

    def test_InputXandY_ReshapeY(self):
        x1 = np.array([1,2,3])
        y1 = np.array([4,5,6])
        y2 = np.array([[4],[5],[6]])
        elm = hpelm.ELM()
        _,y1p = elm.data_loader(x1,y1)
        assert y2.shape == y1p.shape
        assert np.allclose(y2,y1p)

    def test_ClassificationY_CreateTargets(self):
        x1 = np.array([1,2,3,4])
        y1 = np.array([1,1,2,3])
        y2 = np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1]])
        elm = hpelm.ELM("classification")
        _,y1p = elm.data_loader(x1,y1,training=True)
        assert np.allclose(y1p, y2)
        
    def test_ClassificationY_KeepCorrectTargets(self):
        x1 = np.array([1,2,3])
        y1 = np.array([[0,1],[1,0],[1,0]])
        elm = hpelm.ELM("classification")
        _,y1p = elm.data_loader(x1,y1)
        self.assertTrue(np.allclose(y1p, y1))
        
    def test_ClassificationMulticlass_ReturnMulticlassPredictions(self):
        x = np.array([1,2,3,4])
        y = np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1]])
        elm = hpelm.ELM("classification")
        elm.train(x,y)
        yh = elm.predict(x)
        self.assertEqual(yh.shape, y.shape)
        self.assertEqual(yh.min(), 0)        
        self.assertEqual(yh.max(), 1)        
        
    def test_InputX_ZeroMeanUnitVar(self):
        x1 = np.array([1,2,3,10])
        y1 = np.array([1,2,3,4])
        elm = hpelm.ELM()
        xp,_ = elm.data_loader(x1,y1,training=True)
        xp = xp[:,:-1]  # remove bias
        self.assertAlmostEqual(xp.mean(), 0)
        self.assertAlmostEqual(xp.std(), 1)
        
    def test_InputX_StoreMeanVar(self):
        x1 = np.array([1,2,3,10])
        y1 = np.array([1,2,3,4])
        elm = hpelm.ELM()
        elm.data_loader(x1,y1,training=True)  # remove bias
        self.assertAlmostEqual(x1.mean(0), elm.Xmean)
        self.assertAlmostEqual(x1.std(0), elm.Xstd)
        
    def test_TrainAndTestX_TestXNormalized(self):
        """Test is well normalized if taken from the same distribution.
        """
        x = (np.random.randn(1000) - 1)*2
        xtr = x[:500]
        xts = x[500:]
        ytr = np.random.rand(500)
        elm = hpelm.ELM()
        elm.data_loader(xtr, ytr, training=True)
        x2 = elm.data_loader(xts)
        x2 = x2[:,:-1]  # remove bias
        self.assertLess(x2.mean(), 0.15)
        self.assertLess(np.abs(x2.std()-1), 0.15)
        
    def test_InputTextCSV_Loads(self):
        d = os.path.join(os.path.dirname(__file__), "../datasets/iris")
        xf = os.path.join(d, "iris_data.txt")        
        x_csvf = os.path.join(d, "iris_data_comma.txt")        
        elm = hpelm.ELM()
        x = elm.data_loader(xf)
        x_csv = elm.data_loader(x_csvf, delimiter=",")
        self.assertTrue(np.allclose(x, x_csv))

    """
    def test_SetBatch_ReturnIterator(self):
        x = np.random.rand(10)
        elm = hpelm.ELM()
        x2 = elm.data_loader(x, batch=7)
        self.assertAlmostEqual(x[:7], x2.next())
        self.assertAlmostEqual(x[7:], x2.next())
    """
    

class TestInit(TestCase):

    def test_InitDefault_LinearAndTanhNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        elm.train(X,Y)
        self.assertIn(np.copy, elm.ufunc)
        self.assertIn(np.tanh, elm.ufunc)

    def test_InitLinear_LinearNeurons(self):
        elm = hpelm.ELM()
        elm.add_neurons(1, ['lin',3])
        self.assertIn(np.copy, elm.ufunc)
        
    def test_InitNone_LinearNeurons(self):
        elm = hpelm.ELM()
        elm.add_neurons(1, [None,3])
        self.assertIn(np.copy, elm.ufunc)
        
    def test_InitTanh_TanhNeurons(self):
        elm = hpelm.ELM()
        elm.add_neurons(1, ['tanh',3])
        self.assertIn(np.tanh, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
        
    def test_InitCustom_CustomNeurons(self):
        fun = np.sin
        elm = hpelm.ELM()
        elm.add_neurons(1, [fun,3])
        self.assertIn(fun, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
                
    def test_InitBiasW_CorrectW(self):
        elm = hpelm.ELM()
        W0 = np.random.rand(1,3)
        B0 = np.random.rand(1,3)
        elm.add_neurons(1, ['tanh',3,W0,B0])
        assert np.allclose(elm.W, np.vstack((W0,B0)))        
        
    def test_InitTwoBiasW_CorrectW(self):
        elm = hpelm.ELM()
        W0 = np.random.rand(1,3)
        B0 = np.random.rand(1,3)
        W1 = np.random.rand(1,2)
        B1 = np.random.rand(1,2)
        elm.add_neurons(1, [np.sin,3,W0,B0], ['tanh',2,W1,B1])
        W0correct = np.vstack((W0,B0))
        W1correct = np.vstack((W1,B1))
        Wcorrect = np.hstack((W0correct, W1correct))
        assert np.allclose(elm.W, Wcorrect)        
        
    def test_InitFunctionsList_InitCorrectly(self):
        elm = hpelm.ELM()
        func = [np.copy, np.tanh, np.sin]
        elm.add_neurons(1, [func,3])
        self.assertIn(np.copy, elm.ufunc)
        self.assertIn(np.tanh, elm.ufunc)
        self.assertIn(np.sin, elm.ufunc)
        
    def test_LinearNeurons_IdentityW(self):
        elm = hpelm.ELM()
        elm.add_neurons(3, ['lin',2])
        W = elm.W
        self.assertTrue(np.allclose(W[:-1,:], np.eye(3)))
        self.assertEqual(W.shape[0], 3+1)  # including bias
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        