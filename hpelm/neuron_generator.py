# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 23:12:35 2014

@author: Anton
"""
import numpy as np
from scipy.special import expit as sigm




def gen_neurons(d, xmean, xstd, neurons):
    """Generate desired neurons with random projection matrix and transformation functions.

    :param d: dimensionality of inputs (without added bias)
    :param ufunc: desired neuron transformation function, not vectorized
    :param nn: number of neurons to add
    :param W: projection matrix if given
    :param B: bias if given
    """    

    # extract parameters
    nn = neurons[0]
    ufunc = neurons[1]
    W = None
    B = None
    if len(neurons) > 2: W = neurons[2]
    if len(neurons) > 3: B = neurons[3]

    # a list of functions is given
    if hasattr(ufunc, '__iter__'):
        nn = len(ufunc)
    # linear function
    elif (ufunc == 'lin') or (ufunc is None):
        ufunc = [np.copy]*nn
        # copy data values, normalized to bias and variance
        if W is None:
            nn = min(d,nn)
            W = np.eye(d) / xstd.reshape(-1,1)
            B = -np.dot(W.T, xmean)
            W = W[:,:nn]
            B = B[:nn]
    # hyperbolic tangent function
    elif ufunc == 'tanh':
        ufunc = [np.tanh]*nn
    # sigmoid function
    elif ufunc in ['sigm','sigmoid']:
        ufunc = [sigm]*nn
    # any other function
    else:
        assert hasattr(ufunc, '__call__'), "Neuron transformation function type not understood: %s" % (type(ufunc))
        ufunc = [ufunc]*nn

    # generate W and B if they are not given
    if W is None:
        W = np.random.randn(d, nn) / (d**0.5)
        W = W / xstd.reshape(-1,1)  # normalize variance
    if B is None:
        B = -np.dot(W.T, xmean)  # normalize bias

    W = np.vstack((W, B))
    return ufunc, W
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        