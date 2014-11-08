# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:01:02 2014

@author: akusok
"""

import numpy as np


n = 12
d = 6
nn = 5

X1 = np.random.randint(200,255,size=(n,d/3))
X2 = np.random.randint(0,255,size=(n,d/3))
X3 = np.random.randint(1,5,size=(n,d/3))
X = np.hstack((X1,X2,X3)) 
#print X
mean = X.mean(0)
std = X.std(0)

W = np.random.randn(d,nn) / d**0.5
W = W / std.reshape(-1,1)
bias = -np.dot(W.T, mean)

H = X.dot(W) + bias
print H
print H.std()
print ((np.abs(H) < 5)*(np.abs(H)>0.2)).astype(np.int)

"""
from matplotlib import pyplot as plt
a = np.linspace(-5,5,10000)
b = np.tanh(a)
plt.plot(a,b,'-r')
plt.show()
"""