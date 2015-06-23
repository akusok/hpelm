# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np

from slfn import SLFN


class HPELM(SLFN):
    """High performance GPU and parallel Extreme Learning Machine.
    """

    # inherited  def __init__(self, inputs, outputs):
    # inherited  def _checkdata(self, X, T):
    # inherited  def add_neurons(self, number, func, W=None, B=None):
    # inherited  def project(self, X):
    # inherited  def predict(self, X):
    # inherited  def save(self, model):
    # inherited  def load(self, model):

    def __init__(self, X, T, *args, **kvargs):
        """Universal contructor of ELM model with model structure selection.
        
        :param X: input data matrix
        :param T: target data matrix
        :param Xmean: vector of mean value of X for normalization  !!! hpelm
        :param Xstd: vector of srd of X for normalization  !!! hpelm
        
        Model structure selection (exclusive, choose one)
        :param "V": use validation set
        :param "CV": use cross-validation
        :param "MCCV": use Monte-Carlo cross-validation
        :param "LOO": use leave-one-out validation
        
        Additional validation parameters
        :param Xv: validation data X ("V")
        :param Tv: validation targets T ("V")
        :param k: number of splits ("CV", "MCCV")
        :param n: number of repetitions ("MCCV")

        Ranking of hidden neurons
        :param "HQ": use Hannan-Quinn criterion
        # no OP-ELM
        
        System setup
        :param "classification"/"c": build ELM for classification
        :param "multiclass"/"mc": build ELM for multiclass classification
        :param "adaptive"/"ad": build adaptive ELM for non-stationary model
        :param "batch": batch size for adaptive ELM (sliding window step size)
        
        """

        print X.shape
        print T.shape
        print "start args"
        for arg in args:
            print arg
        print "start kvargs"
        for kv, arg in kvargs.items():
            print kv, arg
        print "end (kv)args"
        
        N, inputs = X.shape
        _, targets = T.shape
        super(ELM, self).__init__(inputs, targets)

    def train(self, X, T):
        """Learn a model to project inputs X to targets T.

        :param X: - matrix of inputs
        :param T: - matrix of targets
        """
        assert len(self.neurons) > 0, "Add neurons before training ELM"
        X, T = self._checkdata(X, T)
        H = self.project(X)
        self.Beta = np.linalg.pinv(H).dot(T)
































































    '''  COPY OF OLD PRUNING METHODS
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
            
            # get hidden layer outputs
            H = np.dot(X1,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            H,T1 = semi_Tikhonov(H,T1,self.Tmean)  # add Tikhonov regularization
            
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
        


    def prune_op2(self, X, T, norm=1, batch=10000, delimiter=" "):
        """Prune ELM with a more recent implementation of MRSR.
        
        :param norm: - check numpy.linalg.norm(X, <norm>)
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
            
            # get hidden layer outputs
            H = np.dot(X1,self.W)
            for i in xrange(H.shape[1]):
                H[:,i] = self.ufunc[i](H[:,i])
            H,T1 = semi_Tikhonov(H,T1,self.Tmean)  # add Tikhonov regularization

            # get ranking of neurons in that batch
            # this MRSR2 is a class, with <.rank> attribute and <.new_input()> method
            M = mrsr2(H, T1, norm)
            M.new_input()
            M.new_input()
            
            # select best number of neurons
            MSE = press(H[:, M.rank], T1, self.classification, self.multiclass)
            R_opt = M.rank
            early_stopping = int(nn/10) + 1  # early stopping if no improvement in 10% neurons
            last_improvement = 0
            for i in range(3, nn):
                last_improvement += 1
                M.new_input()
                mse1 = press(H[:, M.rank], T1, self.classification, self.multiclass)
                if mse1 < MSE * delta:
                    MSE = mse1
                    R_opt = M.rank
                    last_improvement = 0
                elif last_improvement > early_stopping:  # early stopping if MSE raises 
                    break
            rank = R_opt
            del M            
            
            # save number of neurons and their ranking information
            nfeats.append(len(rank)) 
            # first selected neuron gets weight 2, last one gets weight 1
            neurons[rank] += np.linspace(2,1,num=len(rank))

        # combine neuron ranking
        nfeats = np.round(np.mean(nfeats))
        neurons = np.argsort(neurons)[::-1][:nfeats]  # sorting in descending order
        
        # update ELM parameters and re-calculate B
        self.W = self.W[:,neurons]
        self.ufunc = [self.ufunc[j] for j in neurons]
        self.train(X, T, batch=batch, delimiter=delimiter)
    '''
































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        