# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:05:03 2014

@author: akusoka1
"""

import numpy as np
from time import time

n = 60000
k = 784
d = 10

H = np.random.randn(n,k)
Y = np.random.randn(n,d)

t = time()
W = H.T.dot(H)
t = time() - t
print "runtime: %.2f"  % t


print "done"
