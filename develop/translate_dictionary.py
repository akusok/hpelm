# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:30:23 2014

@author: akusok
"""


import numpy as np


a = [["a","a","b","c","c"],["c","a","a","a","a"]]
b = np.random.randint(0,3,size=(10,5))


dictA = {"a":0, "b":1, "c":2}
vA = np.vectorize(lambda x : dictA[x])
#print vA(a)


dictB = {0:10, 1:20, 2:30}
vB = np.vectorize(lambda x : dictB[x])
#print vB(b)


decode = {n: l for l,n in dictA.items()}
vD = np.vectorize(lambda x : decode[x])
#print vD(b)


####################################################
print "final test"
f = np.random.randint(0,3,size=(12,))
f = vD(f)
print "f: ", f


C = len(set(f))
Cval = list(set(f))
temp = np.eye(C)
dictF = {Cval[i] : temp[i] for i in range(C)}
def vF(data):
    return np.vstack([dictF[val] for val in data])
print vF(f)


f2 = np.array([[1,0,0],
               [0,1,0],
               [0,0,1]])
un_dictF = {np.argmax(v): k for k,v in dictF.items()}
def un_vF(data):
    return [un_dictF[i] for i in np.argmax(data, 1)]
print un_vF(f2)























