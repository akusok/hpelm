'''
Created on Aug 18, 2014

@author: akusoka1
'''

import numpy as np
from numpy.linalg import lstsq

from elm_error import ELMError



class ELM(object):
    """Vanilla Extreme Learning Machine, high-level wrapper.
    """

    def __init__(self, inputs, targets):
        """Create ELM with fixed input and target dimensionality.
        
        :param inputs: dimensionality of inputs (number of data features).
        :param outputs: dimensionality of targets (number of simultaneously predicted values).
        """
        self.inputs = np.int32(inputs)
        self.targets = np.int32(targets)
        self.W = None
        self.bias = None
        self.ufunc = None
        self.B = None
        
        
    def add_neurons(self, count, ufunc, W=None, bias=None):
        """Add neurons of a given type to the model.
        
        Can specify W input weight vector and B scalar bias. 
        :param count: number of neurons to add
        :param ufunc: transformation function of those neurons, can use "None" for identity function
        :param W: weight matrix for input-to-hidden layer
        :param bias: biases for hidden layer
        """
        # check and compile W
        if W is None:
            W = np.random.randn(self.inputs, count) / (self.inputs**0.5)
        else:
            assert len(W.shape) == 2
        
        # check and compile biases
        if bias is None:
            bias = np.random.randn(count)
        else:
            assert len(bias.shape) == 1
        
        # check and compile nonlinear functions list
        if ufunc is None:
            ufunc = [np.copy] * count  # identity
        else:
            ufunc = [ufunc] * count
        
        # update network structure
        if self.W is None:
            self.W = W
        else:
            self.W = np.hstack((self.W, W))
        
        if self.bias is None:
            self.bias = bias
        else:
            self.bias = np.hstack((self.bias, bias))

        if self.ufunc is None:
            self.ufunc = ufunc
        else:
            self.ufunc.extend(ufunc)


    def project_H(self, X):            
        """Calculate hidden layer output H.
        """
        Hp = X.dot(self.W) + self.bias
        H = np.empty(Hp.shape)
        for i in range(len(self.ufunc)):
            H[:,i] = self.ufunc[i](Hp[:,i])
        return H


    def project_HHT(self, X, T):
        H = self.project_H(X)
        HH = np.dot(H.T, H)
        HT = np.dot(H.T, T)
        return HH, HT


    def project_HB(self, X):
        H = self.project_H(X)
        HB = np.dot(H, self.B)
        return HB

    
    def _solve(self, H, T):
        """Solves HB=Y problem using least square solver.
        """
        self.B = lstsq(H, T)[0]

    
    def train(self, X, T):
        """Wrapper for training ELM.
        
        :param X: Training inputs
        :type X: 2-d matrix
        :param T: Training targets
        :type T: 1-d or 2-d matrix
            
        Trains the ELM model with input features and corresponding desired outputs.
        Number of inputs and outputs must be the same.
        
        If no neurons are added to the model, adds 'tanh' neurons automatically.
        """        

        # make necessary checks
        if X.shape[0] != T.shape[0]:
            raise ELMError("Invalid number of targets: %d expected, %d found" % (X.shape[0], T.shape[0]))
                    
        HH, HT = self.project_HHT(X,T)
        self._solve(HH, HT)
    
    
    def run(self, X):
        """Get predictions using a trained or loaded ELM model.
        
        :param X: input data
        :rtype: predictions Th
        """
        Th = self.project_HB(X)
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

