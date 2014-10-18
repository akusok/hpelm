# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:55:42 2014

@author: akusok
"""

import numpy as np


def ELM_Naive(X, T, nn, Xs=None, Ts=None):
    """Naively simple implementation of ELM.
    """
 
    n,d = X.shape
    mean = np.mean(X,0)
    std = np.std(X,0)
    X = (X - mean) / std        

    W = np.random.randn(d, nn) / (d**0.5)
    bias = np.random.randn(nn)
    
    H = np.dot(X,W) + bias
    H = np.tanh(H)
    B = np.dot(np.linalg.pinv(H), T)

    if Xs is not None:
        Xs = (Xs - mean) / std
    else:
        Xs = X
        Ts = T
    
    Th = np.tanh(np.dot(Xs,W)).dot(B)
    return np.mean((Th-Ts)**2)




