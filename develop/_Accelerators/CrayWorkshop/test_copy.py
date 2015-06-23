# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:05:03 2014

@author: akusoka1
"""

import numpy as np
import os
#from f_apply import f_apply

"""
n = 100000
k = 2000
H = np.ones((n,k))
f = np.random.randint(0,2,size=(k,)).astype(np.int32)
f_apply(H,f)
"""

'''
D_GE_TRI - inverse from triangular factorization
D_GE_LS - solve over- or under-determined linear system
D_GE_SVD - SVD of a matrix

'''




H = np.random.randn(100,5)
Y = np.random.randn(100,2)

H.astype(np.float64).tofile("H.bin")
Y.astype(np.float64).tofile("Y.bin")
os.system("gcc my_svd.c -o mysvd -framework accelerate && ./mysvd")


from scipy.linalg.lapack import dgelss
v,x,s,rank,work,info = dgelss(H,Y)

#print v.shape, x.shape, s.shape, rank, work, info

W = x
W2 = np.fromfile("W.bin", dtype=np.float64).reshape(100,2)

W3 = np.linalg.pinv(H).dot(Y)

print np.linalg.norm(H.dot(W[:5,:]) - Y)
print np.linalg.norm(H.dot(W2[:5,:]) - Y)
print np.linalg.norm(H.dot(W3) - Y)

print "done"












"""

XP = np.linalg.pinv(X)

U,s,V = np.linalg.svd(X, full_matrices=False)

print "U"
print U

print "s"
print s

print "V"
print V

print "done"
raise IOError

X2 = U.dot( (np.diag(s)).dot(V) )
XP2 = ( V.T.dot(np.diag(1/s)) ).dot(U.T)

V2 = np.fromfile("A.bin", dtype=np.float64).reshape(100,100).T * -1

print V2[:3,:3]
print V[:3,:3]


s.astype(np.float64).tofile("s.bin")


assert np.allclose(X, X2)
print "svd correct"

assert np.allclose(XP.dot(X), np.eye(100))
print "Pseudoinverse correct"

assert np.allclose(XP2.dot(X), np.eye(100))
print "Pseudoinverse via SVD correct"

"""


