# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:27:13 2015

@author: akusok
"""

from magma_solver import GPUSolver
import numpy as np
import sys
from time import time


@profile
def s_cpu(data, n, nrhs):
    XX = np.zeros((n,n))
    XT = np.zeros((n,nrhs))
    for X,T in data:
        t = time()
        XX = XX + X.T.dot(X)
        XT = XT + X.T.dot(T)
        print "added batch in %.2fs" % (time()-t)
    t = time()
    P = np.linalg.inv(XX)
    print "solution in %.2fs" % (time()-t)
    Beta = P.dot(XT)
    return Beta

    
@profile
def s_gpu(data, n, nrhs):
    s = GPUSolver(n, nrhs)
    for X,T in data:
        s.add_data(X, T)
    B = s.solve()
    return B


@profile
def try1():
    n = int(sys.argv[1])
    nrhs = int(sys.argv[2])
    k = int(sys.argv[3])
    
    data = []
    for _ in range(k):
        n1 = int(n*0.7)
        X = np.random.rand(n1,n).astype(np.float64)
        T = np.random.rand(n1,nrhs).astype(np.float64)
        data.append((X,T))

    B = s_gpu(data, n, nrhs)
    Beta = s_cpu(data, n, nrhs)
    
    print np.linalg.norm(B - Beta)



   
try1()
print "Works!"
