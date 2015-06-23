# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:18:35 2015

@author: akusok
"""

import numpy as np
from numpy.linalg import pinv, inv, norm
from numpy import dot

#@profile
def prepare():
    N = 125
    nn = 5
    o = 2
    H = np.random.randn(N, nn)
    W = np.random.randn(nn, o)
    T = dot(H,W) + np.random.randn(N,o)*(0.01/6)

    # basic
    HH = H.T.dot(H)
    HT = H.T.dot(T)
    P = inv(HH)
    B = P.dot(HT)
    
    # 10-N    
    j = 10
    H0 = H[j:]
    T0 = T[j:]
    HH0 = H0.T.dot(H0)
    HT0 = H0.T.dot(T0)
    P0 = inv(HH0)
    B0 = P0.dot(HT0)
    
    
    # N substract 10
    H1 = H[:j]
    T1 = T[:j]
    #a = inv(np.eye(j) + H1.dot(P0).dot(H1.T))            # +
    #P1 = P0 - P0.dot(H1.T).dot(a).dot(H1).dot(P0)          # -
    #B1 = P1.dot(H1.T).dot(T1) + (np.eye(nn) - P1.dot(H1.T).dot(H1)).dot(B0) # + -

    a = inv(np.eye(j) - H1.dot(P).dot(H1.T))            # +
    P1 = P + P.dot(H1.T).dot(a).dot(H1).dot(P)          # -
    #P1 = inv(HH - H1.T.dot(H1))
    B1 = B - P1.dot(H1.T).dot(T1) + P1.dot(H1.T).dot(H1).dot(B) # + -
    

    print norm(B0-B1)

prepare()
print "Done"


