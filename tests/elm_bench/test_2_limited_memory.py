# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 20:27:15 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np


import hpelm



class TestBatchProcessing(TestCase):
    
    
    def test_BatchTrain_Works(self):
        np.random.seed(0)
        X = np.random.randn(50,5)
        Y = np.random.randn(50,2)
        elm = hpelm.ELM()
        elm.train(X,Y,batch=17)
        self.assertIsNotNone(elm.B)

    def test_BatchPredict_Works(self):
        np.random.seed(0)
        X = np.random.randn(50,5)
        Y = np.random.randn(50,2)
        elm = hpelm.ELM()
        elm.train(X,Y)
        Yh = elm.predict(X,batch=13)
        mse = np.mean((Y-Yh)**2)
        self.assertGreater(mse, 0)
 
    def test_BatchLOO_Works(self):
        np.random.seed(0)
        X = np.random.randn(50,5)
        Y = np.random.randn(50,2)
        elm = hpelm.ELM()
        elm.train(X,Y)
        mse = elm.loo_press(X, Y, batch=11)
        self.assertGreater(mse, 0)
















       
        