# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:21:55 2015

@author: akusok
"""

import numpy as np
from slfn import SLFN

X = np.random.rand(100,5)
Y = X.dot(np.random.randn(5,3)) + np.random.randn(100,3)*0.1

slfn = SLFN(5,3)
slfn.add_neurons(6,"lin")
slfn.add_neurons(10,"sigm")
slfn.add_neurons(5,"tanh")
slfn.add_neurons(2,"rbf_l1")
slfn.add_neurons(2,"tanh")
H = slfn.project(X)
print slfn
