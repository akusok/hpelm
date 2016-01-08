"""
Created on Thu Mar 19 16:55:19 2015

@author: akusok
"""

import cython

import numpy as np
cimport numpy as np

cdef extern from "gpu_code.h":
    cdef void solve_corr (int n, int nrhs, double* A, double* B, double* X)
    cdef cppclass GpuSolver:
        GpuSolver( int, int, double* A, double* B) except +
        void add_data( int m, double* X, double* T )
        void get_corr( double* XX, double* XT )
        void solve( double* X )
        void finalize()


cdef class GPUSolver:

    cdef GpuSolver *solverptr
    cdef int n, nrhs
    
    def __init__(self, nn, outs, norm=1E-9):
        self.n = nn
        self.nrhs = outs
        cdef np.ndarray[double, ndim=2, mode="c"] A = np.zeros((self.n, self.n))
        cdef np.ndarray[double, ndim=2, mode="c"] B = np.zeros((self.nrhs, self.n))
        A.ravel()[::self.n+1] += norm
        self.solverptr = new GpuSolver( self.n, self.nrhs, &A[0,0], &B[0,0] )

    def finalize(self):
        self.solverptr.finalize()
        
    def add_data(self,
                 np.ndarray[double, ndim=2, mode="c"] X not None,
                 np.ndarray[double, ndim=2, mode="c"] T not None):
        cdef int m = X.shape[0]
        # seems to require Fortran ordering but in "c" mode
        T = T.T.reshape(m, self.nrhs).copy()
        X = X.T.reshape(m, self.n).copy()
        self.solverptr.add_data( m, &X[0,0], &T[0,0] )
        
    def get_corr(self):
        cdef np.ndarray[double, ndim=2, mode="c"] XX = np.empty((self.n, self.n))
        cdef np.ndarray[double, ndim=2, mode="c"] XT = np.empty((self.nrhs, self.n))
        self.solverptr.get_corr( &XX[0,0], &XT[0,0] )
        XX = np.ascontiguousarray(XX.T)
        XT = np.ascontiguousarray(XT.T)
        return XX, XT
        
    def solve(self):
        cdef np.ndarray[double, ndim=2, mode="c"] B = np.empty((self.nrhs, self.n))
        self.solverptr.solve( &B[0,0] );
        return B.T  # transpose back

        
def gpu_solve(np.ndarray[double, ndim=2, mode="c"] XX not None,
              np.ndarray[double, ndim=2, mode="c"] XT not None,
              double norm):              
    cdef int n = XT.shape[0]
    cdef int nrhs = XT.shape[1]
    cdef np.ndarray[double, ndim=2, mode="c"] B = np.empty((nrhs, n))
    
    XX.ravel()[::n+1] += norm
    XT = XT.T.reshape(n,nrhs)  # transpose and cast to 'c' ordering
    solve_corr(n, nrhs, &XX[0,0], &XT[0,0], &B[0,0])
    B = B.reshape(nrhs,n).T
    return B  # transpose











