# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import lstsq

from .data_loader import batchX, batchT, meanstdX, c_dictT, decode
from .neuron_generator import gen_neurons
from .regularizations import semi_Tikhonov, mrsr
from .error_functions import press


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
        elif "multiclass" in args:
            self.multiclass = True
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
        self.C_dict = None  # dictionary for translating classes to binary representation
        self.cI = 1E-9
        np.set_printoptions(precision=5, suppress=True)



    def add_neurons(self, neurons, N):
        # runs if there is an uninitialized model or more neurons specified
        if (self.W is None) or (len(neurons) > 0):  
            # init W correctly
            if self.W is None: 
                self.W = np.empty((self.inputs+1, 0))
            if len(neurons) == 0:  # basic setup if no neurons are specified
                nn = min(5*self.inputs, int(N**0.5))
                neurons = ((self.inputs, 'lin'), (nn, 'sigm'))
            elif not hasattr(neurons[0], '__iter__'):  # fix neurons not being inside a list
                neurons = [neurons]
            # add neurons of desired type
            for ntype in neurons:
                ufunc, W = gen_neurons(self.inputs, self.Xmean, self.Xstd, ntype)
                self.ufunc.extend(ufunc)  
                self.W = np.hstack((self.W, W))



    def train(self, X, T, batch=10000, delimiter=" ", neurons=[]):
        """Trains ELM, can use any X and T(=Y), and specify neurons.
        
        Neurons: (number, type, [W], [B])
        """        
        
        # get parameters of new data and add neurons
        self.Xmean, self.Xstd = meanstdX(X, batch, delimiter)
        if self.classification: self.C_dict = c_dictT(T, batch)
            
        # get data iterators
        genX, self.inputs, N = batchX(X, batch, delimiter)
        genT, self.targets  = batchT(T, batch, delimiter, self.C_dict)
        self.add_neurons(neurons, N)
        
        # project data
        nn = len(self.ufunc)
        HH = np.zeros((nn, nn))
        HT = np.zeros((nn, self.targets))
        for X,T in zip(genX, genT):
            X,T = semi_Tikhonov(X,T)  # add Tikhonov regularization
            
            # get hidden layer outputs
            H = np.dot(X,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            
            # least squares solution - multiply both sides by H'
            p = float(X.shape[0]) / N
            HH += np.dot(H.T, H)*p
            HT += np.dot(H.T, T)*p
            
        # solve ELM model
        HH += self.cI * np.eye(nn)  # enhance solution stability
        self.B = lstsq(HH, HT)[0]
        #self.B = np.linalg.pinv(HH).dot(HT)
        


    def predict(self, X, batch=10000, delimiter=" "):
        """Get predictions using a trained or loaded ELM model.
        
        :param X: input data
        :rtype: predictions Th
        """
        
        assert self.B is not None, "train this model first"
        genX, inputs, _ = batchX(X, batch, delimiter)
        
        results = []
        for X in genX:        
            assert self.inputs == inputs, "incorrect dimensionality of inputs"
            # project test inputs to outputs
            H = np.dot(X,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            Th1 = H.dot(self.B)  
            # additional processing for classification
            if self.classification:
                Th1 = decode(Th1, self.C_dict)
            results.append(Th1)

        # merge results            
        if isinstance(results[0], np.ndarray):
            Th = np.vstack(results)
        else:
            Th = []  # merge results which are lists of items
            for r1 in results: Th.extend(r1)
                
        return Th
        


    def loo_press(self, X, Y, batch=10000, delimiter=" "):
        """PRESS (Predictive REsidual Summ of Squares) error.
        
        Trick is to never calculate full HPH' matrix.
        """

        MSE = 0
        genX, _, N = batchX(X, batch, delimiter)
        genT, _  =   batchT(Y, batch, delimiter, self.C_dict)

        for X,T in zip(genX, genT):
            H = np.dot(X,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            
            p = float(X.shape[0]) / N
            MSE += press(H, T, self.classification, self.multiclass) * p
        
        return MSE
        
        
    def prune_op(self, X, T, batch=10000, delimiter=" "):
        """Prune ELM as in OP-ELM paper.
        """        
        # get data iterators
        genX, self.inputs, N = batchX(X, batch, delimiter)
        genT, self.targets  = batchT(T, batch, delimiter, self.C_dict)
        
        # project data
        nn = len(self.ufunc)
        delta = 0.95  # improvement of MSE for adding more neurons
        nfeats = []
        neurons = np.zeros((nn,))
        for X1,T1 in zip(genX, genT):
            X1,T1 = semi_Tikhonov(X1,T1)  # add Tikhonov regularization
            
            # get hidden layer outputs
            H = np.dot(X1,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            
            # get ranking of neurons in that batch
            rank = mrsr(H, T1, nn)
            
            # select best number of neurons
            MSE = press(H[:, rank[:2]], T1, self.classification, self.multiclass)
            R_opt = rank[:2]
            early_stopping = int(nn/10) + 1  # early stopping if no improvement in 10% neurons
            last_improvement = 0
            for i in range(3, nn):
                last_improvement += 1
                r = rank[:i]
                mse1 = press(H[:,r], T1, self.classification, self.multiclass)
                if mse1 < MSE * delta:
                    MSE = mse1
                    R_opt = r
                    last_improvement = 0
                elif last_improvement > early_stopping:  # early stopping if MSE raises 
                    break
            r = R_opt
            
            # save number of neurons and their ranking information
            nfeats.append(len(r)) 
            # first selected neuron gets weight 2, last one gets weight 1
            neurons[r] += np.linspace(2,1,num=len(r))

        # combine neuron ranking
        nfeats = np.round(np.mean(nfeats))
        neurons = np.argsort(neurons)[::-1][:nfeats]  # sorting in descending order
        
        # update ELM parameters and re-calculate B
        self.W = self.W[:,neurons]
        self.ufunc = [self.ufunc[j] for j in neurons]
        self.train(X, T, batch=batch, delimiter=delimiter)
        





































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        