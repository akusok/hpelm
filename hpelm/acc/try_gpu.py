# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:27:13 2015

@author: akusok
"""

from gpu_solver import GPUSolver, gpu_solve
import numpy as np
from scipy.linalg import solve
import sys
from time import time


def s_cpu(data, n, nrhs):
    XX = np.zeros((n, n))
    XT = np.zeros((n, nrhs))
    for X, T in data:
        t = time()
        XX += X.T.dot(X)
        XT += X.T.dot(T)
        print "CPU: added batch in %.2fs" % (time()-t)
    t = time()
    Beta = solve(XX, XT, sym_pos=True)
    print "CPU: solution in %.2fs" % (time()-t)
    return Beta


def s_gpu(data, n, nrhs):
    s = GPUSolver(n, nrhs)
    for X, T in data:
        t = time()
        s.add_data(X, T)
        print "GPU: added batch in %.2fs" % (time()-t)
    t = time()
    B = s.solve()
    print "GPU: solution in %.2fs" % (time()-t)
    return B


def test():
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

    print "Norm of CPU-GPU difference:", np.linalg.norm(B - Beta)


if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print "add arguments: number of neurons, number of targets, number of batches of data"
    else:
        test()
    
    print "Works!"





















































