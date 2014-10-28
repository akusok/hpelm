# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np


import hpelm


class TestPreprocess(TestCase):
    
    def test_InputX_ReshapeAddBias(self):
        x1 = [1,2,3]
        x2 = np.array([[1,1],[2,1],[3,1]])
        elm = hpelm.ELM()
        x1p = elm.preprocess(x1)
        assert x2.shape == x1p.shape
        assert np.allclose(x2,x1p)
    

    def test_InputXandY_ReshapeY(self):
        x1 = np.array([1,2,3])
        y1 = np.array([4,5,6])
        y2 = np.array([[4],[5],[6]])
        elm = hpelm.ELM()
        _,y1p = elm.preprocess(x1,y1)
        assert y2.shape == y1p.shape
        assert np.allclose(y2,y1p)

    def test_ClassificationY_CreateTargets(self):
        x1 = np.array([1,2,3,4])
        y1 = np.array([0,1,2,2])
        y2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
        elm = hpelm.ELM("classification")
        print elm.classification
        _,y1p = elm.preprocess(x1,y1)
        print y1p
        print y2
        assert np.allclose(y1p, y2)
        
    def test_ClassificationY_CreateZeroBasedTargets(self):
        x1 = np.array([1,2,3,4])
        y1 = np.array([1,2,3,3])
        y2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
        elm = hpelm.ELM("classification")
        _,y1p = elm.preprocess(x1,y1)
        assert np.allclose(y1p, y2)
        
    def test_ClassificationY_CorrectBinaryClassification(self):
        x1 = np.array([1,2,3])
        y1 = np.array([[0],[1],[1]])
        elm = hpelm.ELM("classification")
        _,y1p = elm.preprocess(x1,y1)
        assert np.allclose(y1p, y1)
        

class TestInit(TestCase):

    def test_InitDefault_LinearAndTanhNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        elm.train(X,Y)
        self.assertIn(np.copy, elm.ufunc)
        self.assertIn(np.tanh, elm.ufunc)

    def test_InitLinear_LinearNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        elm.init(1, ['lin',3])
        self.assertIn(np.copy, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
        
    def test_InitNone_LinearNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        elm.init(1, [None,3])
        self.assertIn(np.copy, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
        
    def test_InitTanh_TanhNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        elm.init(1, ['tanh',3])
        self.assertIn(np.tanh, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
        
    def test_InitCustom_CustomNeurons(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        fun = np.sin
        elm = hpelm.ELM()
        elm.init(1, [fun,3])
        self.assertIn(fun, elm.ufunc)
        self.assertEquals(len(elm.ufunc), 3)
                
    def test_InitBiasW_CorrectW(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        W0 = np.random.rand(1,3)
        B0 = np.random.rand(1,3)
        elm.init(1, ['lin',3,W0,B0])
        assert np.allclose(elm.W, np.vstack((W0,B0)))        
        
    def test_InitTwoBiasW_CorrectW(self):
        X = [1,2,3,4]
        Y = [0,0,1,1]
        elm = hpelm.ELM()
        W0 = np.random.rand(1,3)
        B0 = np.random.rand(1,3)
        W1 = np.random.rand(1,2)
        B1 = np.random.rand(1,2)
        elm.init(1, ['lin',3,W0,B0], ['tanh',2,W1,B1])
        W0correct = np.vstack((W0,B0))
        W1correct = np.vstack((W1,B1))
        Wcorrect = np.hstack((W0correct, W1correct))
        assert np.allclose(elm.W, Wcorrect)        
        
        
        
        
        
        
        
        
        
        
        
        