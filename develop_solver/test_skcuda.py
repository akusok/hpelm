# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:36:46 2015

@author: Anton
"""

from solver_skcuda import SolverSkcuda
import numpy as np
from scipy.optimize import minimize_scalar

import pycuda
from pycuda import gpuarray
from skcuda import linalg, misc, cublas


def play():
    N = 50000
    batch = 3571
    L = 5000
    c = 64
    prec = np.float64
    min_batch = 100

    try:
        linalg.init()
    except:
        print "Cannot load cuSOLVER (probably CUDA version too low), using CPU solver."
        
    n = 0
    dev = misc.init_device(n)
    ctx = misc.init_context(dev)    
    print "GPU compute capability:", misc.get_compute_capability(dev)
        
    HH = np.eye(L, dtype=prec)
    HT = np.zeros((L, c), dtype=prec)


    # check absolute minimum required memory (sanity check)
    try:
        devHH = gpuarray.to_gpu(HH)
        devHT = gpuarray.to_gpu(HT)
        devH = misc.zeros((min_batch*2, L), dtype=prec)
        devT = misc.zeros((min_batch*2, c), dtype=prec)
        del devH
        del devT
    except pycuda._driver.MemoryError:
        print "Not enough GPU memory - use less neurons"
        return


    # actual try memory in GPU algorithms, returns cost for minimizing
    def try_mem(b):
        devH = 0
        devT = 0
        b = int(b)
        try:
            devH = misc.zeros((b*2, L), dtype=prec)
            devT = misc.zeros((b*2, c), dtype=prec)
            cost = -b
        except pycuda._driver.MemoryError:
            cost = -1./b
        finally:
            del devH
            del devT
        return cost
    
    cost = try_mem(batch)
    if np.abs(cost) == batch:
        pass  # can use original batch size
    else:
        # find a maximum usable batch
        print "GPU memory limited - performance may be sub-optimal."
        print "Searching for maximum allowed batch size..."
        res = minimize_scalar(try_mem, bounds=(min_batch, batch/2), method='Bounded', options={'xatol':10})    
        Bold = batch
        batch = max(int(res.x), min_batch)
        print "New batch size: %d  (was %d)" % (batch, Bold)
        print "If an ELM still crashes, try other batch sizes smaller than %d, or update CUDA" % batch



    nbatch = int(np.ceil(N * 1.0 / batch))
    print "%d * %d = %d >= %d" % (batch, nbatch, batch*nbatch, N)
    
    devH = 0
    devT = 0
    
    
    handle = cublas.cublasCreate()
    
    for b in xrange(nbatch):
        istart = b*batch
        istop = min((b+1)*batch, N)

        H = np.random.rand(istop-istart, L).astype(prec)
        T = np.random.rand(istop-istart, c).astype(prec)

        # this guy synchronizes GPU
        devH = None  
        devT = None
        devH = gpuarray.to_gpu(H)
        devT = gpuarray.to_gpu(T)

        # these two will execute asyncronously until "devH = None" on next iteration
        linalg.add_dot(devH, devH, devHH, transa='T')
        linalg.add_dot(devH, devT, devHT, transa='T')

        print b+1, nbatch

    # this guy synchronizes GPU
    devH = None
    devT = None
    cublas.cublasDestroy(handle)




@profile
def run():
    N = 3000
    d = 25
    k = 100
    L = 10000/3
    c = 2

    neurons = [('lin', L, np.random.rand(d, L), np.random.rand(L)),
               ('tanh', L, np.random.rand(d, L), np.random.rand(L)),
               ('sigm', L, np.random.rand(d, L), np.random.rand(L))]

    sl = SolverSkcuda(neurons, c, precision=np.float64)
    for _ in xrange(k):
        X = np.random.rand(N, d)
        T = np.random.rand(N, c)
#        _ = sl._project(X)
        sl.add_batch(X, T)

    sl.solve()     
    Y = sl.predict(X)
    print np.mean((T - Y)**2)





if __name__ == "__main__":
    run()        

    print "Done!"
