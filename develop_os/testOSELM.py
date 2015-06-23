# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:18:35 2015

@author: akusok
"""

import numpy as np
from numpy.linalg import pinv, inv, norm
from numpy import dot

def prepare():
    N = 125
    nn = 5
    o = 2
    H = np.random.randn(N, nn)
    W = np.random.randn(nn, o)
    T = dot(H,W) + np.random.randn(N,o)*(0.01/6)
    return H, T
    
@profile
def runall(H,T):
    _ = inv(H.T.dot(H))
    B = run1(H,T)
    B1 = run2(H,T)
    return B, B1

@profile
def run1(H,T):
    # basic
    HH = H.T.dot(H)
    HT = H.T.dot(T)
    P = inv(HH)
    B = P.dot(HT)
    return B
    
@profile
def run2(H,T):
    # OS
    j = 5
    H0 = H[:-j]
    H1 = H[-j:]
    T0 = T[:-j]
    T1 = T[-j:]
    K0 = dot(H0.T, H0)
    P0 = inv(K0)
    a = inv(np.eye(j) + H1.dot(P0).dot(H1.T))
    P1 = P0 - P0.dot(H1.T).dot(a).dot(H1).dot(P0)
    B0 = P0.dot(H0.T.dot(T0))
    B1 = B0 + P1.dot(H1.T.dot(T1) - H1.T.dot(H1).dot(B0))
    return B1    

h,t = prepare()
b,b1 = runall(h,t)
print norm(b-b1)
print "Done"


