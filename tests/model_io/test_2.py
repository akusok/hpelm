# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:13:53 2014

@author: akusok
"""


from unittest import TestCase
import numpy as np


import hpelm



#  2. Error functions
class TestErrors(TestCase):
    
    def test_LooPress_ReturnsMSE(self):
        x = np.linspace(-1,1,100)
        y = np.sin(x)
        elm = hpelm.ELM()
        elm.train(x,y)
        E = elm.loo_press(x,y)
        assert E < 0.1