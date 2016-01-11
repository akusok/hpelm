# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import numexpr as ne
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
import os

from nnet import SLFN


class parallel_cdist(object):

    def __init__(self, W, kind):
        self.C = W.T
        self.kind = kind

    def __call__(self, X):
        return cdist(X, self.C, self.kind)


class ELM(SLFN):
    """Non-parallel Extreme Learning Machine.
    """

    # inherited  def __init__(self, inputs, outputs):
    # inherited  def _checkdata(self, X, T):
    # inherited  def add_neurons(self, number, func, W=None, B=None):
    # inherited  def project(self, X):
    # inherited  def predict(self, X):
    # inherited  def save(self, model):
    # inherited  def load(self, model):

    def _mp_project(X, W, B, k, kind):
        pcdist = parallel_cdist(W, kind)
        # fix issues with running everything on one core
        os.system("taskset -p 0xff %d >/dev/null" % os.getpid())
        p = Pool(k)
        H0 = p.map(pcdist, np.array_split(X, k, axis=0))
        H0 = np.vstack(H0) / (-2 * (B ** 2))
        p.close()
        return H0

    def project(self, X):
        # assemble global hidden layer output
        H = []
        for func, ntype in self.neurons.iteritems():
            _, W, B = ntype
            k = cpu_count()

            # projection
            if func == "rbf_l2":
                H0 = self._mp_project(X, W, B, k, "sqeuclidean")
            elif func == "rbf_l1":
                H0 = self._mp_project(X, W, B, k, "cityblock")
            elif func == "rbf_linf":
                H0 = self._mp_project(X, W, B, k, "chebyshev")
            else:
                H0 = X.dot(W) + B

            # transformation
            if func == "lin":
                pass
            elif "rbf" in func:
                ne.evaluate('exp(H0)', out=H0)
            elif func == "sigm":
                ne.evaluate("1/(1+exp(-H0))", out=H0)
            elif func == "tanh":
                ne.evaluate('tanh(H0)', out=H0)
            else:
                H0 = func(H0)  # custom <numpy.ufunc>
            H.append(H0)

        H = np.hstack(H)
        return H

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
































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        