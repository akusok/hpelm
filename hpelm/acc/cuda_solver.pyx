"""
Created on Thu Mar 19 16:55:19 2015

@author: akusok
"""

import cython

import numpy as np
cimport numpy as np

cdef extern from "cuda_code.h":
    cdef cppclass CudaSolver:
        CudaSolver( int, int, double* HH, double* HT) except +
        void add_data( int N, double* H, double* T );
        void get_corr( double* HH, double* HT );
        void finalize()


cdef class CUDASolver:

    cdef CudaSolver *solverptr
    cdef int L, C
    
    def __init__(self, nL, nC, norm=1E-9):
        self.L = nL
        self.C = nC
        cdef np.ndarray[double, ndim=2, mode="c"] HH = np.zeros((self.L, self.L))
        cdef np.ndarray[double, ndim=2, mode="c"] HT = np.zeros((self.L, self.C))
        np.fill_diagonal(HH, norm)
        self.solverptr = new CudaSolver( self.L, self.C, &HH[0,0], &HT[0,0] )

    def add_data(self,
                 np.ndarray[double, ndim=2, mode="c"] H not None,
                 np.ndarray[double, ndim=2, mode="c"] T not None):
        cdef int N = H.shape[0]
        T = T.T.reshape(N, self.C).copy()
        H = H.T.reshape(N, self.L).copy()
        self.solverptr.add_data( N, &H[0,0], &T[0,0] )

    def get_corr(self):
        cdef np.ndarray[double, ndim=2, mode="c"] HH = np.empty((self.L, self.L))
        cdef np.ndarray[double, ndim=2, mode="c"] HT = np.empty((self.C, self.L))
        self.solverptr.get_corr( &HH[0,0], &HT[0,0] )
        HH = np.ascontiguousarray(HH.T)
        HT = np.ascontiguousarray(HT.T)
        return HH, HT


    def finalize(self):
        self.solverptr.finalize()
        










