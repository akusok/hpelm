# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 20:27:30 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np


import hpelm



class TestErrorFunctions(TestCase):
    
    
    def test_PressRegression_Works(self):
        np.random.seed(0)
        Y = np.random.randn(50,2)
        W = np.random.rand(2,5)
        X = Y.dot(W) + 0.01 * np.random.randn(50,5)        
        mse = hpelm.press(X,Y,False,False)
        self.assertGreater(mse, 0)


    def test_PressClassification_IgnoresSmallNoise(self):
        np.random.seed(0)
        Y = np.zeros((50,2))
        Y[:25,0] = 1
        Y[25:,1] = 1
        W = np.random.rand(2,5)
        X = Y.dot(W) + 0.01 * np.random.randn(50,5)        
        mse = hpelm.press(X,Y,True,False)
        self.assertEqual(mse, 0)


    def test_PressMulticlass_IgnoresSmallNoise(self):
        np.random.seed(0)
        Y = np.random.randn(50,3) > 0.6        
        W = np.random.rand(3,5)
        X = Y.dot(W) + 0.01 * np.random.randn(50,5)        
        mse = hpelm.press(X,Y,False,True)
        self.assertEqual(mse, 0)


    def test_PressClassification_ReportsMisclassification(self):
        np.random.seed(0)
        Y = np.zeros((50,2))
        Y[:25,0] = 1
        Y[25:,1] = 1
        Y2 = Y.copy()
        Y2[0,0] = 0
        Y2[0,1] = 1
        W = np.random.rand(2,5)
        X = Y2.dot(W) + 0.01 * np.random.randn(50,5)        
        mse = hpelm.press(X,Y,True,False)
        self.assertGreater(mse, 0)


    def test_PressMulticlass_ReportsMisclassification(self):
        np.random.seed(2)
        Y = np.random.randn(50,3) > 0.6        
        Y2 = Y.copy()
        Y2[0,0] = -Y2[0,0]+1
        Y3 = Y2.copy()
        Y3[0,1] = -Y3[0,1]+1
        W = np.random.rand(3,5)
        X2 = Y2.dot(W) + 0.01 * np.random.randn(50,5)        
        X3 = Y3.dot(W) + 0.01 * np.random.randn(50,5)        
        mse2 = hpelm.press(X2,Y,False,True)
        mse3 = hpelm.press(X3,Y,False,True)
        self.assertGreater(mse2, 0)
        self.assertLess(mse2, mse3)







       
        