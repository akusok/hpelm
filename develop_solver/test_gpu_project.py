# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:17:24 2015

@author: akusok
"""

import numpy as np
import pycuda
from pycuda import autoinit, gpuarray, cumath
from skcuda import linalg, misc, cublas
from pycuda.compiler import SourceModule


@profile
def run():
    N = 10000
    d = 25
    L = 9000/3
    k = 1
    precision = np.float32
    X = np.random.rand(N, d).astype(precision)

    neurons = [('lin', L, np.random.rand(d, L).astype(precision), np.random.rand(L).astype(precision)),
               ('tanh', L, np.random.rand(d, L).astype(precision), np.random.rand(L).astype(precision)),
               ('sigm', L, np.random.rand(d, L).astype(precision), np.random.rand(L).astype(precision))]
    L = L*3

    # host
    func = {}
    func['lin'] = lambda a: a
    func['sigm'] = lambda a: 1/(1 + np.exp(a))
    func['tanh'] = lambda a: np.tanh(a)

    for _ in xrange(k):
        H =  np.hstack([func[ftype](np.dot(X, W) + B) for ftype, _, W, B in neurons])

    # device
    linalg.init()

    # move neurons to GPU
    dev_neurons = []
    for ftype, l1, W, B in neurons:
        devW = gpuarray.to_gpu(W)
        devB = gpuarray.to_gpu(B.reshape((1, -1)))
        dev_neurons.append((ftype, l1, devW, devB))

    kernel = """
        __global__ void dev_sigm(%s *a) {
            unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
            a[idx] = 1.0 / ( exp(a[idx]) + 1 );
        }
        """
    kernel = kernel % "double" if precision is np.float64 else kernel % "float"
    mod_sigm = SourceModule(kernel)
    dev_sigm = mod_sigm.get_function("dev_sigm")
    dev_sigm.prepare("P")

    def sigm(a):
        block = a._block
        grid = (int(np.ceil(1.0 * np.prod(a.shape) / block[0])), 1)
        dev_sigm.prepared_call(grid, block, a.gpudata)
        return a

    dev_func = {}
    dev_func['lin'] = lambda a: a
#    dev_func['sigm'] = lambda a: misc.divide(misc.ones_like(a), cumath.exp(a)+1)    
#    dev_func['sigm'] = lambda a: 1.0 / (cumath.exp(a) + 1)
    dev_func['sigm'] = sigm
    dev_func['tanh'] = lambda a: cumath.tanh(a, out=a)
        
    for _ in xrange(k):
        i = 0
        devX = gpuarray.to_gpu(X)
        devH = gpuarray.empty((N, L), dtype=precision)
        for ftype, l1, devW, devB in dev_neurons:
            devH[:, i:i+l1] = dev_func[ftype](misc.add_matvec(linalg.dot(devX, devW), devB, axis=1))
            i += l1
    
    assert np.allclose(H, devH.get())  

##################################################

    # SYRK works for single precision

    handle = cublas.cublasCreate()
    devHH = gpuarray.zeros((L, L), dtype=precision)
    
    k2 = 10
    ctx = pycuda.driver.Context
    for _ in xrange(k2):
        linalg.add_dot(devH, devH, devHH, transa='T')        
        ctx.synchronize()
        if precision is np.float32:
            # this works only for single precision, don't ask why
            cublas.cublasSsyrk(handle, 'L', 'N', L, N, 1, devH.ptr, L, 1, devHH.ptr, L)
        else:
            # double precision alternative
            linalg.add_dot(devH, devH, devHH, transa='T')        
        ctx.synchronize()

    HH2 = devHH.get()
                
    HH = np.dot(H.T, H)

    
    cublas.cublasDestroy(handle)


    HH2 = np.triu(HH2) + np.triu(HH2, k=1).T
    HH2 = HH2 / (2*k2)

    assert np.allclose(HH, HH2)            








if __name__ == "__main__":
    run()

print 'done'



















