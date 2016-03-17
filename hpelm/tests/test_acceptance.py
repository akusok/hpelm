# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:12:41 2014

@author: akusok
"""

from unittest import TestCase
import numpy as np

import hpelm


# noinspection PyArgumentList
class TestAcceptance(TestCase):

    def test_basic_elm_single_machine(self):
        """Just run an ELM with sine function and check training MSE.
        """
        n = 1000
        err = 0.2
        Y = np.linspace(-1, 1, num=n)
        X = np.sin(16 * Y) * Y + np.random.randn(n) * err

        elm = hpelm.ELM(1, 1)
        elm.add_neurons(3, "sigm")
        print(elm.nnet.get_B())
        elm.train(X, Y)
        print(elm.nnet.get_B())
        Yt = elm.predict(X)

        MSE = np.mean((Y - Yt) ** 2)
        self.assertLess(MSE, 0.5)

    def test_xor_one_neuron_solved(self):
        """ELM should be able to solve XOR problem.
        """
        X = np.array([[0, 0],
                      [1, 1],
                      [1, 0],
                      [0, 1]])
        Y = np.array([1, 1, -1, -1])
        for _ in range(100):  # try 100 random initializations to solve XOR problem
            try:
                elm = hpelm.ELM(2, 1)
                elm.add_neurons(1, "sigm")
                elm.train(X, Y)
                Yh = elm.predict(X)
                self.assertGreater(Yh[0], 0)
                self.assertGreater(Yh[1], 0)
                self.assertLess(Y[2], 0)
                self.assertLess(Y[3], 0)
                nn = sum([n[0] for n in elm.nnet.neurons])
                self.assertEqual(nn, 1)  # one neuron in the ELM
                return
            except AssertionError:
                pass
        self.fail("Cannot train 1 neuron to solve XOR problem in 100 re-initializations")
