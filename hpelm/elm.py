# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.special import expit as sigm

from .data_loader import batchX, batchT, meanstdX, c_dictT, decode
from .neuron_generator import gen_neurons


class ELM(object):
    """Non-parallel Extreme Learning Machine.
    """

    def __init__(self, *args):
        """Create ELM of desired kind.
        
        :param regression: type of ELM task, can be regression, classification or timeseries regression.
        :param sparse: set to create an ELM with sparse projection matrix.
        """
        self.classification = False
        self.multiclass = False
        self.regression = False
        if "classification" in args:
            self.classification = True
            print "starting ELM for classification"
        elif "timeseries" in args:
            self.timeseries = True
            print "starting ELM for timeseries"
        else:
            self.regression = True
            print "starting ELM for regression"
                
        # set default argument values
        self.inputs = None
        self.targets = None
        self.N = None
        self.W = None
        self.ufunc = []
        self.B = None
        self.Xmean = 0
        self.Xstd = 1
        self.Tmean = 0
        self.C_dict = None  # dictionary for translating classes to binary representation
        np.set_printoptions(precision=5, suppress=True)


    def train(self, X, T, batch=10000, delimiter=" ", neurons=[]):
        """Trains ELM, can use any X and T(=Y), and specify neurons.
        
        Neurons: (number, type, [W], [B])
        """        
        
        self.Xmean, self.Xstd = meanstdX(X, batch, delimiter)
        if self.classification: 
            self.C_dict = c_dictT(T, batch)
        else:
            self.Tmean,_ = meanstdX(T, batch, delimiter)
        
        # get data iterators
        genX, self.inputs, N  = batchX(X, batch, delimiter)
        genT, self.targets = batchT(T, batch, delimiter, self.C_dict)
        
        # uninitialized model or more neurons 
        if (self.W is None) or (len(neurons) > 0):  
            # init W correctly
            if self.W is None: self.W = np.empty((self.inputs+1, 0))
            if len(neurons) == 0:  # basic setup if no neurons are specified
                nn = min(5*self.inputs, int(N**0.5))
                neurons = ((self.inputs, 'lin'), (nn, 'sigm'))
            elif not hasattr(neurons[0], '__iter__'):  # fix neurons not being in a list
                neurons = [neurons]

            # add neurons of desired type
            for ntype in neurons:
                ufunc, W = gen_neurons(self.inputs, self.Xmean, self.Xstd, ntype)
                self.ufunc.extend(ufunc)  
                self.W = np.hstack((self.W, W))

        X = np.vstack(genX)
        T = np.vstack(genT)
        
        # add semi-Tikhonov regularization: small random noise projected to "zero"
        # "zero" = zero + E[T], otherwise we introduce a bias
        nT = X.shape[0]/10 + 10
        xT = np.random.rand(nT, X.shape[1]-1) * 10E-9
        xT = np.hstack((xT, np.ones((nT,1))))
        X = np.vstack((X, xT))
        T = np.vstack((T, np.tile(self.Tmean, (nT,1))))
        
        H = np.dot(X,self.W)

        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
            
        T = np.dot(H.T, T)                  
        nn = H.shape[1]
        H = np.dot(H.T, H) + 1E-9 * np.eye(nn)
            
        self.B = lstsq(H, T)[0]
        #self.B = np.linalg.pinv(H).dot(T)
        


    def predict(self, X, batch=10000, delimiter=" "):
        """Get predictions using a trained or loaded ELM model.
        
        :param X: input data
        :rtype: predictions Th
        """
        
        assert self.B is not None, "train this model first"
        genX, inputs, _ = batchX(X, batch, delimiter)
        
        X = np.vstack(genX)
        assert self.inputs == inputs, "incorrect dimensionality of inputs"

        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
 
        Th = H.dot(self.B)  
        if self.classification:
            if self.multiclass:
                Th = np.array(Th > 0.5, dtype=np.int)
            else:
                Th = decode(Th, self.C_dict)
        return Th
        

    def loo_press(self, X, Y):
        """PRESS (Predictive REsidual Summ of Squares) error.
        
        Trick is to never calculate full HPH' matrix.
        """
        X = np.vstack(batchX(X)[0])        
        Y = np.vstack(batchT(Y)[0])        
        
        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
                    
        # TROP-ELM stuff with lambda
        lmd = 0
        Ht = H.T
        HtY = Ht.dot(Y)
        P = np.linalg.inv(Ht.dot(H) + np.eye(H.shape[1])*lmd)
        HP = H.dot(P)

        e1 = Y - HP.dot(HtY)  # e1 = Y - HPH'Y
        Pdiag = np.einsum('ij,ji->i', HP, Ht)        
        e2 = np.ones((H.shape[0], )) - Pdiag  # diag(HPH')

        e = e1 / np.tile(e2, (Y.shape[1],1)).T  # PRESS error
        E = np.mean(e**2)  # MSE PRESS        

        return E
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        