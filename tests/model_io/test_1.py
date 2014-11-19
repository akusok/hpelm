# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:13:13 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np
import os

import hpelm


#  1. Load data from files
class TestTrainFromFile(TestCase):
    
    def test_Load_TextFiles(self):
        d = os.path.join(os.path.dirname(__file__), "../../datasets/Unittest-Sine")
        x = os.path.join(d, "sine_x.txt")        
        y = os.path.join(d, "sine_y.txt")        
        elm = hpelm.ELM()
        elm.train(x,y)
        self.assertIsNotNone(elm.B)

    def test_Load_Hdf5Files(self):
        d = os.path.join(os.path.dirname(__file__), "../../datasets/Unittest-Sine")
        x = os.path.join(d, "sine_x.h5")        
        y = os.path.join(d, "sine_y.h5")        
        elm = hpelm.ELM()
        elm.train(x,y)
        self.assertIsNotNone(elm.B)
        
    def test_Load_MatrixAndFileMixedInputs(self):
        d = os.path.join(os.path.dirname(__file__), "../../datasets/Unittest-Sine")
        x = os.path.join(d, "sine_x.h5")        
        y = os.path.join(d, "sine_y.txt")        
        y = np.loadtxt(y)
        elm = hpelm.ELM()
        elm.train(x,y)
        self.assertIsNotNone(elm.B)