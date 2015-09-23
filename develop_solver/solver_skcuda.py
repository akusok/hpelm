# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:10:23 2015

@author: akusok
"""

from solver import Solver
import numpy as np
from scipy.linalg import lapack

from pycuda import gpuarray, cumath
from pycuda.compiler import SourceModule
from skcuda import linalg, misc, cublas, cusolver, cula


class SolverSkcuda(Solver):

    def __init__(self, neurons, c, norm=None, precision=np.float64, nDevice=0):
        """Initialize matrices, functions and GPU stuff.
        """
        assert precision in (np.float32, np.float64), \
            "Only single or double precision (numpy.float32, numpy.float64) are supported"
        super(SolverSkcuda, self).__init__(neurons, c, norm, precision)

        # startup GPU
        self.ctx = misc.init_context(misc.init_device(nDevice))
        try:
            linalg.init()
        except Exception as e:
            print "error initializing scikit-cuda: %s" % e
            print "ignore if toolbox works"

        # create GPU matrices
        self.precision = precision
        self.HH = misc.zeros((self.L, self.L), dtype=precision)
        self.HT = misc.zeros((self.L, self.c), dtype=precision)
        self.HH = linalg.eye(self.L, dtype=precision)
        self.HH *= self.norm

        # precision-dependent stuff
        if precision is np.float64:
            self.posv = lapack.dposv
        else:
            self.posv = lapack.sposv
            self.handle = cublas.cublasCreate()

        # move neurons to GPU
        dev_neurons = []
        for ftype, l1, W, B in self.neurons:
            devW = gpuarray.to_gpu(W)
            devB = gpuarray.to_gpu(B.reshape((1, -1)))
            dev_neurons.append((ftype, l1, devW, devB))
        self.neurons = dev_neurons

        # GPU transformation functions
        self.func['sigm'] = self._dev_sigm()
        self.func['tanh'] = lambda a: cumath.tanh(a)

    def _dev_sigm(self):
        """Compute Sigmoid on GPU for a given array and return array.
        """
        kernel = """
            __global__ void dev_sigm(%s *a) {
                unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
                a[idx] = 1.0 / ( exp(a[idx]) + 1 );
            }
            """
        kernel = kernel % "double" if self.precision is np.float64 else kernel % "float"
        mod_sigm = SourceModule(kernel)
        dev_sigm = mod_sigm.get_function("dev_sigm")
        dev_sigm.prepare("P")

        def sigm(a):
            block = a._block
            grid = (int(np.ceil(1.0 * np.prod(a.shape) / block[0])), 1)
            dev_sigm.prepared_call(grid, block, a.gpudata)
            return a
        return sigm

    def _project(self, X):
        """Projects X to H.
        """
        devX = gpuarray.to_gpu(self.to_precision(X))
        devH = gpuarray.empty((X.shape[0], self.L), dtype=self.precision)
        i = 0
        for ftype, l1, devW, devB in self.neurons:
            devH[:, i:i+l1] = self.func[ftype](misc.add_matvec(linalg.dot(devX, devW), devB, axis=1))
            i += l1
        return devH

    def add_batch(self, X, T):
        """Add a batch of data to an iterative solution.
        """
        devH = self._project(X)
        devT = gpuarray.to_gpu(self.to_precision(T))
        linalg.add_dot(devH, devT, self.HT, transa='T')
        if self.precision is np.float64:
            linalg.add_dot(devH, devH, self.HH, transa='T')
        else:
            cublas.cublasSsyrk(self.handle, 'L', 'N', self.L, X.shape[0], 1, devH.ptr, self.L, 1, self.HH.ptr, self.L)
#        self.ctx.synchronize()  # GPU runs asyncronously without that

    def solve(self):
        """Compute output weights B, with fix for unstable solution.
        """
        HH = self.HH.get()
        HT = self.HT.get()
        _, B, info = self.posv(HH, HT)
        if info > 0:
            print "ELM covariance matrix is not full rank; solving with SVD (slow)"
            print "This happened because you have duplicated or too many neurons"
            HH = np.triu(HH) + np.triu(HH, k=1).T
            B = np.linalg.lstsq(HH, HT)[0]
        B = np.ascontiguousarray(B)
        self.B = gpuarray.to_gpu(B)
        return B

    def get_corr(self):
        """Return current correlation matrices.
        """
        HH = self.HH.get()
        HT = self.HT.get()
        HH = np.triu(HH) + np.triu(HH, k=1).T
        return HH, HT

    def predict(self, X):
        """Predict a batch of data.
        """
        assert self.B is not None, "Solve the task before predicting"
        devH = self._project(X)
        devY = linalg.dot(devH, self.B)
        return devY.get()


def run():
    k = 1
    N = 100
    d = 25
    L = 100/3
    c = 2
    precision = np.float64

    neurons = [('lin', L, np.random.rand(d, L), np.random.rand(L)),
               ('tanh', L, np.random.randn(d, L), np.random.rand(L)),
               ('sigm', L, np.random.randn(d, L), np.random.rand(L))]
    L = L*3

    sl = SolverSkcuda(neurons, c, norm=1E-15, precision=precision)
    for j in xrange(k):
        X = np.random.rand(N, d)
        T = np.random.rand(N, c)
        sl.add_batch(X, T)
        print j

    sl.solve()
    Y1 = sl.predict(X)
    print "scikit-cuda", np.mean((T - Y1)**2)


if __name__ == "__main__":
    run()




        # nothing works
#        temp = linalg.svd(self.HH)
#        temp = linalg.cho_solve(self.HH, self.HT)
#        print dir(linalg.cula)
#        temp = linalg.eig(self.HH)
#        temp = linalg.inv(self.HH)
#        temp = linalg.pinv(self.HH)

        # cuSOLVER SVD is 10 times slower than CPU and GPU does nothing, don't ask why...
#        cs_ctx = cusolver.cusolverDnCreate()
#        lwork = cusolver.cusolverDnSgesvd_bufferSize(cs_ctx, self.L, self.L)
#        print lwork
#        devHH = gpuarray.to_gpu(np.asfortranarray(self.HH.get()))
#        devS = gpuarray.zeros((self.L,), dtype=self.precision)
#        devU = gpuarray.zeros((self.L, self.L), dtype=self.precision)
#        devVt = gpuarray.zeros((self.L, self.L), dtype=self.precision)
#        devWrk = gpuarray.zeros((lwork,), dtype=self.precision)
#        dinfo = gpuarray.to_gpu(np.array([0], dtype=np.int))
#        cusolver.cusolverDnSgesvd(cs_ctx, 'A', 'A', self.L, self.L, devHH.gpudata, 
#                                  self.L, devS.gpudata, devU.gpudata, self.L, devVt.gpudata, self.L,
#                                  devWrk.gpudata, lwork, lwork, dinfo.gpudata)
#        print dinfo
#        U = devU.get().T
#        S = devS.get()
#        Vt = devVt.get().T
#        HH1 = self.HH.get()
#        HH2 = U.dot(np.diag(S)).dot(Vt)
#        print np.mean((HH1 - HH2)**2)
#        cusolver.cusolverDnDestroy(cs_ctx)
#        temp2 = np.linalg.svd(HH1)

#        # CULA solver
#        cula.culaInitialize()
#        devHH = gpuarray.to_gpu(np.asfortranarray(self.HH.get()))
#        devS = gpuarray.zeros((self.L,), dtype=self.precision)
#        devU = gpuarray.zeros((self.L, self.L), dtype=self.precision)
#        devVt = gpuarray.zeros((self.L, self.L), dtype=self.precision)
#        
#        # this crap is 0.6 speed of np.linalg.lstsq(), just forget it
#        cula.culaDeviceSgesvd('A', 'A', self.L, self.L, devHH.gpudata, self.L,
#                              devS.gpudata, devU.gpudata, self.L, devVt.gpudata, self.L)
#        U = devU.get().T
#        S = devS.get()
#        Vt = devVt.get().T
#        HH1 = self.HH.get()
#        HH2 = U.dot(np.diag(S)).dot(Vt)
#        print np.mean((HH1 - HH2)**2)
#
#        temp = np.linalg.svd(HH1)
#
#        cula.culaShutdown()













