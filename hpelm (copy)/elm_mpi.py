'''
Created on Aug 18, 2014

@author: akusoka1
'''

import numpy as np
from numpy.linalg import lstsq
from mpi4py import MPI

from elm_error import ELMError
from elm import ELM


class ELM_MPI(ELM):
    """Distributed Extreme Learning Machine, works with MPI.
    """

    def __init__(self, inputs, targets):
        """Create ELM with fixed input and target dimensionality.
        
        :param inputs: dimensionality of inputs (number of data features).
        :param outputs: dimensionality of targets (number of simultaneously predicted values).
        """
        
        ELM.__init__(self, inputs, targets) 
        self.comm = MPI.COMM_WORLD        
        self.drank = self.comm.Get_rank()
        self.dsize = self.comm.Get_size()

        
    def add_neurons(self, count, ufunc, W=None, bias=None):
        """Add neurons of a given type to the model.

        Runs only on root.        
        """
        if self.drank == 0:
            ELM.add_neurons(self, count, ufunc, W, bias)    
        else:
            self.W = None
            self.ufunc = None
            self.bias = None
        self.W = self.comm.bcast(self.W, root=0)
        self.ufunc = self.comm.bcast(self.ufunc, root=0)
        self.bias = self.comm.bcast(self.bias, root=0)
            

    def _distribute(self, X, T=None):            
        """Distribute data and targets across all nodes.
        """
        # distribute parameters
        if self.drank == 0:
            n = X.shape[0]
            batch = n / self.dsize
            if self.dsize*batch < n:
                batch += 1
            if T is None:  # should I distribute targets
                has_T = False
            else:
                has_T = True
        else:
            n = None
            batch = None
            has_T = None
        n = self.comm.bcast(n, root=0)
        batch = self.comm.bcast(batch, root=0)
        has_T = self.comm.bcast(has_T, root=0)
        d = self.inputs
        nn = self.W.shape[1]
        
        # distribute input data
        if self.drank == 0:
            # split and distribute data X
            for rec in range(1,self.dsize):
                self.comm.Isend([X[batch*rec: batch*(rec+1)], MPI.DOUBLE], dest=rec, tag=1)            
            X = X[:batch]  # take first batch for itself
        else:
            X = np.empty((batch, d), dtype=np.float64)
            # we can receive smaller array into larger buffer
            self.comm.Recv([X, MPI.DOUBLE], source=0, tag=1)

        # distribute target data
        if has_T:
            if self.drank == 0:
                # split and distribute targets T
                for rec in range(1,self.dsize):
                    self.comm.Isend([T[batch*rec: batch*(rec+1)], MPI.DOUBLE], dest=rec, tag=2)            
                T = T[:batch]  # take first batch for itself
            else:
                T = np.empty((batch, self.targets), dtype=np.float64)
                # we can receive smaller array into larger buffer
                self.comm.Recv([T, MPI.DOUBLE], source=0, tag=2)

        # connect to different endings            
        return X, T, batch, n, nn


    def project_H(self, X):
        """Compute projection matrix distributively.
        """                 
        # start distributed computation
        X, _, batch, n, nn = self._distribute(X)                 

        # compute
        H1 = X.dot(self.W) + self.bias
        for i in xrange(nn):
            H1[:,i] = self.ufunc[i](H1[:,i])
            
        # return matrix H
        H = np.empty((batch*self.dsize, nn))
        self.comm.Gather([H1, MPI.DOUBLE], [H, MPI.DOUBLE]) 
        if self.drank == 0:
            H = H[:n]
            return H


    def project_HHT(self, X, T):
        """Compute parts of least squares solution of linear system.
        
        Computes H=XW, HH = H'H and HT = H'T. Takes much less space.
        """                 
        # start distributed computation
        X, T, batch, n, nn = self._distribute(X,T)  

        # compute
        HT1 = np.zeros((nn,self.targets))
        HH1 = np.zeros((nn,nn))

        niter = X.shape[0]/nn  # number of mini-batches in local X
        if niter*nn < X.shape[0]: niter += 1
        for i in xrange(niter):
            H1 = X[nn*i:nn*(i+1)].dot(self.W) + self.bias
            for j in xrange(nn):
                H1[:,j] = self.ufunc[j](H1[:,j])
            T1 = T[nn*i:nn*(i+1)]
            HH1 += np.dot(H1.T, H1)
            HT1 += np.dot(H1.T, T1)

        #'''            
        tH1 = X.dot(self.W)
        for i in xrange(nn):
            tH1[:,i] = self.ufunc[i](tH1[:,i])
        tHT1 = np.dot(tH1.T, T)
        tH1 = np.dot(tH1.T, tH1)
        #'''           
           
        # gather matrices
        HT = np.empty(HT1.shape)
        self.comm.Reduce([HT1, MPI.DOUBLE], [HT, MPI.DOUBLE], op=MPI.SUM, root=0) 
        HH = np.empty(HH1.shape)
        self.comm.Reduce([HH1, MPI.DOUBLE], [HH, MPI.DOUBLE], op=MPI.SUM, root=0) 
        if self.drank == 0:
            return HH, HT
        else:
            return None, None


    def project_HB(self, X):
        """Run ELM distrubutively.
        """                 
        # start distributed computation
        X, _, batch, n, nn = self._distribute(X)                 
        if self.drank > 0:
            self.B = None
        self.B = self.comm.bcast(self.B, root=0)

        # compute
        H1 = X.dot(self.W) + self.bias
        for i in xrange(nn):
            H1[:,i] = self.ufunc[i](H1[:,i])
        
        # compute ELM output
        Y1 = np.dot(H1, self.B)
        
        # combine ELM output
        Y = np.empty((batch*self.dsize, self.B.shape[1]))
        self.comm.Gather([Y1, MPI.DOUBLE], [Y, MPI.DOUBLE]) 
        if self.drank == 0:
            Y = Y[:n]
            return Y

    
    def _solve(self, H, T):
        """Solves HB=Y problem using least square solver.
        """
        self.B = lstsq(H, T)[0]


    def train(self, X, T):
        """Wrapper for training ELM.
        """                
        if self.drank == 0:
            # make necessary checks
            if X.shape[0] != T.shape[0]:
                raise ELMError("Invalid number of targets: %d expected, %d found" % (X.shape[0], T.shape[0]))
                    
        HH, HT = self.project_HHT(X,T)
        if self.drank == 0:
            self._solve(HH, HT)
    
    
    def run(self, X):
        """Get predictions using a trained or loaded ELM model.
        
        :param X: input data
        :rtype: predictions Th
        """
        Th = self.project_HB(X)
        if self.drank == 0:        
            return Th
    










































    
    def get_model(self):
        """Returns all parameters of the ELM model.
        
        :rtype: python dictionary
        
        Model does not require specific classes to load and edit, unlike pickled class instances.
    
        Neuron format: list of ['input weights vector', 'bias scalar', 'transformation function as numpy.ufunc']
        """
        model = {}
        model['inputs'] = self.inputs
        model['targets'] = self.targets
        model['neurons'] = self.neurons
        return model
    
    
    def set_model(self, model):
        """Loads ELM model from a Python dictionary.
        
        :param model: ELM model parameters
        :type model: python dictionary
        """
        self.inputs = model['inputs']
        self.targets = model['targets']
        self.neurons = model['neurons']

