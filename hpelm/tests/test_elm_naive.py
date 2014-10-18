# -*- coding: utf-8 -*-


from unittest import TestCase
import numpy as np

import hpelm

class TestElmNaive(TestCase):
    
    def test_xor(self):
        """ELM should be able to solve XOR problem.
        """
        X = np.array([[0,0], [1,1], [1,0], [0,1]])
        Y = np.array([1,1,-1,-1])
        for _ in range(100):
            try:
                Yh = hpelm.ELM_Naive(X,Y)
                self.assertGreater(Yh[0], 0)
                self.assertGreater(Yh[1], 0)
                self.assertLess(Y[2], 0)
                self.assertLess(Y[3], 0)
                return
            except:
                pass
        self.fail("Cannot train 1 neuron to solve XOR problem in 100 re-initializations")        