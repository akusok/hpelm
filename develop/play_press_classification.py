# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 19:09:30 2014

@author: akusok
"""


import numpy as np
from numpy import linalg
np.set_printoptions(precision=5, suppress=True)



def run():
    #Y = np.random.rand(30,2)
    N = 50
    Y = np.zeros((N,2))
    Y[:15,0] = 1
    Y[15:,1] = 1
    
    W = np.random.randn(2,5)
    Y2 = Y.copy()
    n = np.random.randint(1,9)
    for _ in range(n):
        k = np.random.randint(0,N)
        Y2[k] = -Y2[k]+1
    X = Y2.dot(W) + np.random.randn(N,5)*0.1
    
    B = linalg.pinv(X).dot(Y)
    Yh = X.dot(B)
    err = np.mean((Y - Yh)**2)
    #print np.argmax(Y,1)
    #print np.argmax(Yh,1)
    #print "train", err
    
    
    N = X.shape[0]
    C = linalg.inv(np.dot(X.T, X))
    P = X.dot(C)
    W = C.dot(X.T).dot(Y)        
    D = np.ones((N,)) - np.einsum('ij,ji->i', P, X.T)        
    e = (Y - X.dot(W)) / D.reshape((-1,1))
    MSE = np.mean(e**2)
    
    #print 'press', MSE
    
    
    ###################################################
    ###  best classification error here  ###    
    
    N = X.shape[0]
    C = linalg.inv(np.dot(X.T, X))
    P = X.dot(C)
    W = C.dot(X.T).dot(Y)        
    D = np.ones((N,)) - np.einsum('ij,ji->i', P, X.T)        
    e = (Y.argmax(1) - X.dot(W).argmax(1)) /  D.T
    MCE = np.mean(e**2)  
    #print 'class', MCE
    ###################################################
    
    e = Y.argmax(1) - X.dot(W).argmax(1)
    MCE2 = np.mean(e**2)
    #print 'clas2', MCE2

    return MSE, MCE, MCE2



def run_multiclass():
    #Y = np.random.rand(30,2)
    N = 20
    Y = np.random.rand(N,3)
    Y = np.array(Y > 0.6, dtype=np.int)
    
    W = np.random.randn(3,5)
    Y2 = Y.copy()
    for _ in range(np.random.randint(1,5)):
        k = np.random.randint(0,N)
        for _ in range(np.random.randint(0,3)):
            j = np.random.randint(0,3)
            Y2[k,j] = -Y2[k,j]+1
    X = Y2.dot(W) + np.random.randn(N,5)*0.1
    
    N = X.shape[0]
    C = linalg.inv(np.dot(X.T, X))
    P = X.dot(C)
    W = C.dot(X.T).dot(Y)        
    D = np.ones((N,)) - np.einsum('ij,ji->i', P, X.T)        
    e = (Y - X.dot(W)) / D.reshape((-1,1))
    MSE = np.mean(e**2)
    
    #print 'press', MSE
    
    
    ###################################################
    ###  best classification error here  ###    
    
    N = X.shape[0]
    C = linalg.inv(np.dot(X.T, X))
    P = X.dot(C)
    W = C.dot(X.T).dot(Y)        
    D = np.ones((N,)) - np.einsum('ij,ji->i', P, X.T)        

    e = ((Y>0.5) - (X.dot(W)>0.5)) /  D.reshape((-1,1))
    MCE = np.mean(e**2)  
    #print 'class', MCE
    ###################################################
    
    e = ((Y>0.5) - (X.dot(W)>0.5))
    MCE2 = np.mean(e**2)
    #print 'clas2', MCE2

    return MSE, MCE, MCE2






s = [0,0,0,0,0]
N = 10000
for _ in range(N):
    m,m2,m3 = run_multiclass()
    s[0] += m
    s[1] += np.abs(m2-m)
    s[2] += m2-m
    s[3] += np.abs(m3-m)
    s[4] += m3-m
s = np.array(s) * 100 / N

print "MSE, |MSE-MCE|, MSE-MCE, |MSE-MCE2|, MSE-MCE2"
print s










