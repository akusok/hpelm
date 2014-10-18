# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:29:34 2014

@author: akusok
"""

import numpy as np
from time import time

#@profile
def run():

    n = 25000
    d = 4000
    a = np.random.rand(d,n)
    
    f = [np.tanh]*n    
    
    b = np.tanh(a)    
    
    b = np.empty(a.shape)
    for i in xrange(n):
        b[:,i] = f[i](a[:,i])
    
    f2 = np.array(f)
    b = np.empty(a.shape)
    for i in xrange(n):
        b[:,i] = f2[i](a[:,i])
    
    f2 = np.array(f)
    b = a.copy()
    for i in xrange(n):
        f2[i](b[:,i], out=b[:,i])

    f3 = np.random.rand(n)
    f3 = [np.tanh if f3[i] < 0.5 else np.copy for i in xrange(n)]
    b = np.empty(a.shape)
    for i in xrange(n):
        b[:,i] = f3[i](a[:,i])
    
    b = np.copy(a)
    f4 = [np.copy]*n    
    b = np.empty(a.shape)
    for i in xrange(n):
        b[:,i] = f4[i](a[:,i])
    

@profile
def run2():
    """Check multiply-and-add function
    """
    n = 1000
    a = np.random.rand(n,n)
    b = np.random.rand(n,n)
    c = np.random.rand(n,n)
    d = np.random.rand(n,n)

    d = a.dot(b) + c  # 27.3

    d += b.dot(c)  # 20.2
    
    
    x = np.random.rand(100*n,n)    
    w = np.random.rand(n,n)
    
    h = x.dot(w)                    #
    h = np.dot(h.T, h)              # 40.1
    
    h2 = np.zeros((n,n))
    for i in range(100):
        h3 = x[n*i:n*(i+1)].dot(w)  #
        h2 += np.dot(h3.T, h3)      # 43.7
    
    assert(np.allclose(h,h2))
    
run2()
    




























































    
    
    