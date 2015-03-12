# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 20:36:25 2015

@author: akusok
"""

import numpy as np


@profile
def run():
    
    N = 50000
    d = 500
    nn = 9000
    X = np.random.rand(N,d)
    W = np.random.rand(d,nn)
    
    H0 = X.dot(W)
    del H0
    
    H1 = np.empty((N,nn))
    for i in xrange(nn):
        H1[:,i] = X.dot(W[:,i])


    print "done"


run()    