 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import lstsq

from .data_loader import batchX, batchT, c_dictT, decode
from .regularizations import semi_Tikhonov
from .error_functions import press
from .mrsr import mrsr
from .mrsr2 import mrsr2
from .elm import ELM


class ELM_SMALL(ELM):
    """Single-machine Extreme Learning Machine with model selection.
    """

    def __init__(self, *args):
        """Create ELM of desired kind.
        """
        super(ELM_SMALL, self).__init__(*args)
        self.batch = -1  # batch < 0 means no batch processing


    def project(self, X, delimiter=" "):
        """Projects inputs to a hidden layer.
        """        
        X, self.inputs, N = batchX(X, self.batch, delimiter)
        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
        return H


    def train(self, X, T, delimiter=" ", neurons=[]):
        """Trains ELM, can use any X and T(=Y), and specify neurons.
        
        Neurons: (number, type, [W], [B])
        """        
        
        # get data
        if self.classification: 
            self.C_dict = c_dictT(T, self.batch)
        X, self.inputs, N = batchX(X, self.batch, delimiter)
        T, self.targets  = batchT(T, self.batch, delimiter, self.C_dict)

        # get parameters of new data and add neurons
        self.Xmean = X.mean(0)        
        self.Xstd = X.std(0)                
            
        # get mean value of targets
        if self.classification or self.multiclass:
            self.Tmean = np.zeros((self.targets,))  # for any classification
        else:
            self.Tmean = T.mean(0)

        self.add_neurons(neurons, N)
        
        # project data
        nn = len(self.ufunc)
        HH = np.zeros((nn, nn))
        HT = np.zeros((nn, self.targets))
                    
        # get hidden layer outputs
        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
        H,T = semi_Tikhonov(H,T, self.Tmean)  # add Tikhonov regularization

        # least squares solution - multiply both sides by H'
        p = float(X.shape[0]) / N
        HH += np.dot(H.T, H)*p
        HT += np.dot(H.T, T)*p

        # solve ELM model
        HH += self.cI * np.eye(nn)  # enhance solution stability
        self.B = lstsq(HH, HT)[0]
        #self.B = np.linalg.pinv(HH).dot(HT)
        


    def predict(self, X, delimiter=" "):
        """Get predictions using a trained or loaded ELM model.
        
        :param X: input data
        :rtype: predictions Th
        """
        
        assert self.B is not None, "train this model first"
        X, inputs, _ = batchX(X, self.batch, delimiter)
        
        assert self.inputs == inputs, "incorrect dimensionality of inputs"
        # project test inputs to outputs
        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])

        Th = H.dot(self.B)  
        # additional processing for classification
        if self.classification:
            Th = decode(Th, self.C_dict)

        return Th
        


    def loo_press(self, X, Y, delimiter=" "):
        """PRESS (Predictive REsidual Summ of Squares) error.
        
        Trick is to never calculate full HPH' matrix.
        """

        MSE = 0
        X, _, N = batchX(X, self.batch, delimiter)
        T, _  =   batchT(Y, self.batch, delimiter, self.C_dict)

        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
        
        MSE = press(H, T, self.classification, self.multiclass)
        return MSE
        
        
        
    def prune_op(self, X, T, delimiter=" "):
        """Prune ELM as in OP-ELM paper.
        """        
        # get data iterators
        X, self.inputs, N = batchX(X, self.batch, delimiter)
        T, self.targets  = batchT(T, self.batch, delimiter, self.C_dict)
        
        # project data
        nn = len(self.ufunc)
        delta = 0.95  # improvement of MSE for adding more neurons
            
        # get hidden layer outputs
        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
        H,T = semi_Tikhonov(H,T,self.Tmean)  # add Tikhonov regularization
            
        # get ranking of neurons in that batch
        rank = mrsr(H, T, nn)
            
        # select best number of neurons
        MSE = press(H[:, rank[:2]], T, self.classification, self.multiclass)
        R_opt = rank[:2]
        early_stopping = int(nn/10) + 1  # early stopping if no improvement in 10% neurons
        last_improvement = 0
        for i in range(3, nn):
            last_improvement += 1
            r = rank[:i]
            mse1 = press(H[:,r], T, self.classification, self.multiclass)
            if mse1 < MSE * delta:
                MSE = mse1
                R_opt = r
                last_improvement = 0
            elif last_improvement > early_stopping:  # early stopping if MSE raises 
                break
                
        # update ELM parameters and re-calculate B
        self.W = self.W[:,R_opt]
        self.ufunc = [self.ufunc[j] for j in R_opt]
        self.train(X, T)
        


    def prune_op2(self, X, T, norm=1, delimiter=" "):
        """Prune ELM with a more recent implementation of MRSR.
        
        :param norm: - check numpy.linalg.norm(X, <norm>)
        """        
        # get data iterators
        X, self.inputs, N = batchX(X, self.batch, delimiter)
        T, self.targets  = batchT(T, self.batch, delimiter, self.C_dict)
        
        # project data
        nn = len(self.ufunc)
        delta = 0.95  # improvement of MSE for adding more neurons

        # get hidden layer outputs
        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
        H,T = semi_Tikhonov(H,T,self.Tmean)  # add Tikhonov regularization

        # get ranking of neurons in that batch
        # this MRSR2 is a class, with <.rank> attribute and <.new_input()> method
        M = mrsr2(H, T, norm)
        M.new_input()
        M.new_input()
        
        # select best number of neurons
        MSE = press(H[:, M.rank], T, self.classification, self.multiclass)
        R_opt = M.rank
        early_stopping = int(nn/10) + 1  # early stopping if no improvement in 10% neurons
        last_improvement = 0
        for i in range(3, nn):
            last_improvement += 1
            M.new_input()
            mse1 = press(H[:, M.rank], T, self.classification, self.multiclass)
            if mse1 < MSE * delta:
                MSE = mse1
                R_opt = M.rank
                last_improvement = 0
            elif last_improvement > early_stopping:  # early stopping if MSE raises 
                break
        del M            
                
        # update ELM parameters and re-calculate B
        self.W = self.W[:,R_opt]
        self.ufunc = [self.ufunc[j] for j in R_opt]
        self.train(X, T)

































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        