# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:21:00 2014

@author: akusok
"""

import numpy as np
from multiprocessing import Pool

@profile
def run(n,d):

    X = np.random.randn(n,d)
    W = np.random.rand(d,3*d)
    f = []

    ident = lambda x:x
    f.extend([ident] * d)
    f.extend([np.tanh] * (2*d))

    Hp = X.dot(W)
    Hp = np.array(Hp, order="F")
    H = np.empty(Hp.shape)

    for i in range(len(f)):
        H[:,i] = f[i](Hp[:,i])

    H2 = np.vstack([f[i](Hp[:,i]) for i in xrange(len(f))]).T


#@profile
def testid(n):
    X = np.random.rand(n,n)    
    
    id1 = lambda x:x  # this is always faster!!!
    id2 = np.frompyfunc(lambda x:x, 1, 1)
    
    id1(X[:])
    id2(X[:])
    
    
    
    

if __name__ == "__main__":
    #run(1000,10)
    run(1000,100)
    #run(1000,1000)
    #run(100,1000)
    #run(10,1000)
    #testid(1000)








