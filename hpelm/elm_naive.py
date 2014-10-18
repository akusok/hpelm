# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:55:42 2014

@author: akusok
"""

import numpy as np


def ELM_Naive(X, T, nn=None, Xs=None, Ts=None, classification=False):
    """Naively simple implementation of ELM.
    """
 
    n,d = X.shape
    if nn is None:
        nn = int(d**1.5 + 1);
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
    
    Th = np.tanh(np.dot(Xs,W) + bias).dot(B)
    if classification:
        Ts = np.argmax(Ts,1)
        Th = np.argmax(Th,1)
        return float(np.sum(Th==Ts))/Th.shape[0]
    else:        
        return np.mean((Th-Ts)**2)




