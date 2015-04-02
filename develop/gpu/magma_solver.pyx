# distutils: language = c++
# distutils: sources = Rectangle.cpp

"""
Created on Thu Mar 19 16:55:19 2015

@author: akusok
"""

import cython

import numpy as np
cimport numpy as np

#cdef extern void gpu_solve (int n, int nrhs, double* A, double* B, double* X)

cdef extern from "gpu_solver.h":
    cdef cppclass GpuSolver:
        GpuSolver( int, int ) except +
        void add_data( int m, double* X, double* T )
        void solve( double* X )


cdef class GPUSolver:

    cdef GpuSolver *solverptr
    cdef int n, nrhs
    
    def __init__(self, nn, outs):
        self.n = nn
        self.nrhs = outs
        self.solverptr = new GpuSolver( self.n, self.nrhs )

    def __del__(self):
        del self.solver
        
    def add_data(self,
                 np.ndarray[double, ndim=2, mode="c"] X not None,
                 np.ndarray[double, ndim=2, mode="c"] T not None):
        cdef int m = X.shape[0]
        # seems to require Fortran ordering but in "c" mode
        T = T.T.reshape(m, self.nrhs).copy()
        X = X.T.reshape(m, self.n).copy()
        self.solverptr.add_data( m, &X[0,0], &T[0,0] )
        
    #def solve(self,
    #          np.ndarray[double, ndim=2, mode="c"] B not None):
    def solve(self):
        cdef np.ndarray[double, ndim=2, mode="c"] B = np.empty((self.nrhs, self.n))
        self.solverptr.solve( &B[0,0] );
        return B.T  # transpose back
        
 









#def solve(np.ndarray[double, ndim=2, mode="c"] X not None,
#          np.ndarray[double, ndim=2, mode="c"] T not None,
#          np.ndarray[double, ndim=2, mode="c"] B not None):
#    cdef int n = T.shape[0]
#    cdef int nrhs = T.shape[1]

#    T = T.T.reshape(n,nrhs)  # transpose and cast to 'c' ordering
#    gpu_solve(n, nrhs, &X[0,0], &T[0,0], &B[0,0])
#    return B.reshape(nrhs,n).T  # transpose











