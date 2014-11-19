# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:14:22 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
from scipy.special import expit as sigm

import hpelm


#  4. Model initialization
class TestInit(TestCase):

    def test_InitDefault_LinearAndSigmNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        elm.train(X,Y)
        self.assertIn(np.copy, elm.ufunc)
        self.assertIn(sigm, elm.ufunc)

    def test_InitLinear_LinearNeurons(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        nr1 = [3,'lin']
        elm.train(x,y,neurons=nr1)
        self.assertIn(np.copy, elm.ufunc)
        
    def test_InitNone_LinearNeurons(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        nr1 = [3,None]
        elm.train(x,y,neurons=nr1)
        self.assertIn(np.copy, elm.ufunc)
        
    def test_InitTanh_TanhNeurons(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        nr1 = [3,'tanh']
        elm.train(x,y,neurons=nr1)
        self.assertIn(np.tanh, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)

    def test_InitSigmSigmoid_SigmoidNeurons(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        nr1 = [1,'sigm']
        nr2 = [1,'sigmoid']
        elm.train(x,y,neurons=(nr1,nr2))
        self.assertIs(sigm, elm.ufunc[0])
        self.assertIs(sigm, elm.ufunc[1])
        
    def test_InitCustom_CustomNeurons(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        fun = np.sin
        nr1 = [3, fun]
        elm.train(x,y,neurons=nr1)
        self.assertIn(fun, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
                
    def test_InitBiasW_CorrectW(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        W0 = np.random.rand(1,3)
        B0 = np.random.rand(1,3)
        nr1 = [3,'tanh',W0,B0]
        elm.train(x,y,neurons=nr1)
        assert np.allclose(elm.W, np.vstack((W0,B0)))        
        
    def test_InitTwoBiasW_CorrectW(self):
        x = np.array([2,3,4,5])
        y = np.array([1,2,3,2])
        elm = hpelm.ELM()
        W0 = np.random.rand(1,3)
        B0 = np.random.rand(1,3)
        W1 = np.random.rand(1,2)
        B1 = np.random.rand(1,2)
        nr1 = [3,np.sin,W0,B0]
        nr2 = [2,'tanh',W1,B1]
        elm.train(x,y,neurons=(nr1,nr2))
        W0correct = np.vstack((W0,B0))
        W1correct = np.vstack((W1,B1))
        Wcorrect = np.hstack((W0correct, W1correct))
        assert np.allclose(elm.W, Wcorrect)        
        
    def test_InitFunctionsList_InitCorrectly(self):
        x = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
        y = np.array([1,2,3,4])
        elm = hpelm.ELM()
        func = [np.copy, np.tanh, np.sin]
        neuron = (3, func)        
        elm.train(x,y,neurons=(neuron))
        self.assertIn(np.copy, elm.ufunc)
        self.assertIn(np.tanh, elm.ufunc)
        self.assertIn(np.sin, elm.ufunc)
        
    def test_LinearNeurons_IdentityW(self):
        x = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
        y = np.array([1,2,3,4])
        elm = hpelm.ELM()
        neuron = (3,'lin')
        elm.train(x,y,neurons=[neuron])
        W = elm.W
        self.assertEqual(W.shape[0], 3+1)  # including bias
        # all non-diagonal elements are zero
        W = W[:-1]  # remove bias
        np.fill_diagonal(W,0)  # remove diagonal
        self.assertTrue(np.allclose(W, np.zeros((3,3))))
        
    def test_RetrainAddMoreNeurons_GotAdditionalNeurons(self):
        x = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
        y = np.array([1,2,3,4])
        nr1 = [2,'lin']
        nr2 = [3,'tanh']
        elm = hpelm.ELM()
        elm.train(x,y,neurons=(nr1))
        self.assertIn(np.copy, elm.ufunc)
        self.assertEqual(2, len(elm.ufunc))
        elm.train(x,y,neurons=(nr2))
        self.assertIn(np.tanh, elm.ufunc)
        self.assertEqual(5, len(elm.ufunc))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    