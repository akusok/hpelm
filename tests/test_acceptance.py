# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
import os

import hpelm


class TestAcceptance(TestCase):
    
    def test_basic_elm_single_machine(self):
        """Just run an ELM with sine function and report LOO MSE.
        """
        n = 1000
        err = 0.2
        Y = np.linspace(-1,1,num=n)
        X = np.sin(16*Y)*Y + np.random.randn(n)*err

        elm = hpelm.ELM()
        elm.train(X,Y)
        Yt = elm.predict(X)
        
        MSE = np.mean((Y - Yt)**2)        
        
        self.assertLess(MSE, 0.5)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        