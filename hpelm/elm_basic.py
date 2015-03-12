# -*- coding: utf-8 -*-
"""Basic sigmoidal ELM, no normalization or other tricks.
Created on Sat Oct 18 16:55:42 2014

@author: akusok
"""

import numpy as np
from scipy.special import expit as sigm


def ELM_Basic(X, T, nn, Xs=None, Ts=None, classification=False):
    """Most basic implementation of ELM.
    """

    # prepare data and model 
    n,d = X.shape
    W = np.random.randn(d, nn) * 3**0.5
    bias = np.random.randn(nn)
    
    H = np.dot(X,W) + bias  # random projection
    H = np.tanh(H)  # non-linear transformation

    B = np.dot(np.linalg.pinv(H), T)  # linear regression

    #HH = H.T.dot(H) + 1E-3*np.eye(H.shape[1])
    #HT = H.T.dot(T)    
    #B = np.linalg.lstsq(HH, HT)[0]

    # test ELM model
    if Xs is None:  # training error
        Xs = X
        Ts = T
    
    Th = np.tanh(np.dot(Xs,W) + bias).dot(B)
    
    if classification:
        Ts = np.argmax(Ts,1)
        Th = np.argmax(Th,1)
        MSE = 1 - np.sum(Th==Ts)*1.0/Th.shape[0]  # 1 - accuracy
    else:        
        MSE = np.mean((Th-Ts)**2)
        
    return Th, MSE




