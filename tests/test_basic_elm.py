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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        