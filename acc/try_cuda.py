# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:01:05 2015

@author: akusok
"""

from cuda_solver import CUDASolver
import numpy as np
from scipy.linalg import solve
import sys
from time import time, sleep


    
    
    
if __name__ == "__main__":
    N = 5
    L = 3
    C = 2
    
    s = CUDASolver(L, C)
#    H = np.random.rand(N, L)
#    T = np.random.rand(N, C)
#
#    i = 3
#    for j in xrange(i):
#        print j
#        s.add_data(H, T)
    HH, HT = s.get_corr()

#    HT2 = H.T.dot(T)
#    HH2 = H.T.dot(H)
#
#    HH = HH + np.triu(HH, k=1).T
#
#    print np.abs(HH - HH2*i).mean()
#    print np.abs(HT - HT2*i).mean()

    s.finalize()

    print "Works!"


    


















































