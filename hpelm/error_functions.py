# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 18:34:22 2014

@author: akusok
"""
import numpy as np
import scipy as sp


def press(H, T, classification, multiclass):
    """According to Momo's article, fast version with no L2-regularization.
    
    Extended case for multiple outputs, 'W' is 2-dimensional.
    """        
    X = H
    N = X.shape[0]
    C = np.linalg.inv(np.dot(X.T, X))
    P = X.dot(C)
    W = C.dot(X.T).dot(T)        
    D = np.ones((N,)) - np.einsum('ij,ji->i', P, X.T)        
    if classification:
        e = (T.argmax(1) - X.dot(W).argmax(1)) /  D.T  # mis-classification error
    elif multiclass:
        e = ((T>0.5) - (X.dot(W)>0.5)) /  D.reshape((-1,1))  # multi-class mis-classification
    else:
        e = (T - X.dot(W)) / D.reshape((-1,1))  # regression error
    MSE = np.mean(e**2)
    return MSE


def press_L2(H, T, lmd, classification, multiclass):
    """According to Momo's article.
    
    Extended case for multiple outputs, 'W' is 2-dimensional.
    """      
    X = H
    N = X.shape[0]
    U,S,V = np.linalg.svd(X, full_matrices=False)
    A = np.dot(X, V.T)
    B = np.dot(U.T, T)
    
    # function for optimization
    def lmd_opt(lmd, S, A, B, U, N):    
        Sd = S**2 + lmd
        C = A*(S/Sd)
        P = np.dot(C, B)
        D = np.ones((N,)) - np.einsum('ij,ji->i', C, U.T)
        if classification:
            e = (T.argmax(1) - P.argmax(1)) /  D.T  # mis-classification error
        elif multiclass:
            e = ((T>0.5) - (P>0.5)) /  D.reshape((-1,1))  # multi-class mis-classification
        else:
            e = (T - P) / D.reshape((-1,1))  # regression error
        MSE = np.mean(e**2)
        return MSE
    
    res = sp.optimize.minimize(lmd_opt, lmd, args=(S,A,B,U,N), method="Powell")
    if not res.success:
        print "Lambda optimization failed:  (using basic results)"
        print res.message
        MSE = lmd_opt(lmd, S, A, B, U, N)    
        lmd = 0
    else:
        MSE = res.fun
        lmd = res.x

    return MSE, lmd