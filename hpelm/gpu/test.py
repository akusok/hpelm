# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:27:13 2015

@author: akusok
"""

from magma_solver import GPUSolver, gpu_solve
import numpy as np
from scipy.linalg import solve
import sys
from time import time


@profile
def s_cpu(data, n, nrhs):
    XX = np.zeros((n, n))
    XT = np.zeros((n, nrhs))
    for X, T in data:
        t = time()
        XX = XX + X.T.dot(X)
        XT = XT + X.T.dot(T)
        print "added batch in %.2fs" % (time()-t)
    t = time()
    Beta = solve(XX, XT, sym_pos=True)
    print "solution in %.2fs" % (time()-t)
    return Beta


@profile
def s_gpu(data, n, nrhs):
    s = GPUSolver(n, nrhs)
    for X, T in data:
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
        X = np.random.rand(n1, n).astype(np.float64)
        T = np.random.rand(n1, nrhs).astype(np.float64)
        data.append((X, T))

    B = s_gpu(data, n, nrhs)
    Beta = s_cpu(data, n, nrhs)

    print np.linalg.norm(B - Beta)


def getcorr():
    n = 4
    nrhs = 2
    norm = 0.03
    X = np.random.rand(3*n, n).astype(np.float64)
    T = np.random.rand(3*n, nrhs).astype(np.float64)
    s = GPUSolver(n, nrhs, norm)
    s.add_data(X, T)

    XX = X.T.dot(X) + np.eye(n)*norm
    XT = X.T.dot(T)

    xx1, xt1 = s.get_corr()
    assert np.allclose(XX, xx1)
    assert np.allclose(XT, xt1)
    print "check passed"


@profile
def lol():
    n = int(sys.argv[1])
    X = np.ones((n,n))
    norm = 0.1
    X = X + np.eye(n)*norm
    print X[:3,:3]
    X.ravel()[::n+1] += norm
    print X[:3,:3]
    

def solve_independent():
    n = int(sys.argv[1])
    nrhs = int(sys.argv[2])
    X = np.random.rand(3*n, n).astype(np.float64)
    T = np.random.rand(3*n, nrhs).astype(np.float64)
    T = X.T.dot(T)    
    X = X.T.dot(X)
    
    B = gpu_solve(X, T, 1E-9)
    #b1 = np.linalg.pinv(X).dot(T)
    #print "equal result: ", np.allclose(B, b1)



try1()
#getcorr()
#solve_independent()
#lol()
print "Works!"





















































