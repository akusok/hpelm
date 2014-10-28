# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import lstsq



class ELM(object):
    """Non-parallel Extreme Learning Machine.
    """

    def __init__(self, *args):
        """Create ELM of desired kind.
        
        :param regression: type of ELM task, can be regression, classification of timeseries regression.
        :param sparse: set to create an ELM with sparse projection matrix.
        """
        self.classification = False
        self.timeseries = False        
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
        
        if "sparse" in args:
            self.sparse = True
        else:
            self.sparse = False
        
        # set default argument values
        self.inputs = None
        self.targets = None
        self.W = None
        self.ufunc = None
        self.B = None


    def preprocess(self, X, Y=None):
        """Check the correctness of input data, also add bias to X.
        """
        if type(X) is not np.ndarray:
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((-1,1))
        assert len(X.shape) == 2, "X must be 1- or 2-dimensional array"
        
        n = X.shape[0]
        X = np.hstack((X, np.ones(n).reshape((-1,1))))
        
        if Y is None:
            return X
            
        if type(Y) is not np.ndarray:
            Y = np.array(Y)
        if len(Y.shape) == 1:        
            Y = Y.reshape((-1,1))
        assert len(Y.shape) == 2, "Y must be 1- or 2-dimensional array"
        assert Y.shape[0] == n, "X and Y must have equal number of samples"

        # translate class indexes into binary code for classification
        if self.classification:
            # binary classification
            if (Y.shape[1] == 1) and (set(Y.flat) == set([0,1])):
                return X,Y
                
            # one-out-of-many classification
            if (set(Y.flat) == set([0,1])) and (np.all(Y.sum(1) == 1)):
                return X,Y
                
            # integer classes
            if (Y.shape[1] == 1) and (np.allclose(Y, Y.astype(np.int))):
                setY = set(Y.flat)
                c = len(setY)  # number of classes
                # set zero-based class indexes if they are 1-based
                if setY == set(range(1,c+1)): 
                    Y = Y - 1
                    setY = set(Y.flat)
                if setY == set(range(c)):                    
                    Y2 = np.zeros((Y.shape[0], c))
                    Y2[range(n), Y.flat] = 1
                    Y = Y2
                    return X,Y
                    
            # targets have incorrect format for classification
            assert False, "Required class indexes are 1..[number_of_classes]"

        return X,Y


    def init(self, inputs, *args):
        """Add neurons of a given type to the model.
        
        Can specify W input weight vector and B scalar bias. 
        :param count: number of neurons to add
        :param ufunc: transformation function of those neurons, can use "None" for identity function
        :param W: weight matrix for input-to-hidden layer
        :param bias: biases for hidden layer
        """

        # set or check dimensionality
        if self.inputs is None:
            self.inputs = inputs
        else:
            assert self.inputs == inputs, "model currently has different number of inputs"
            
        # set some defaults  
        d = self.inputs
        nn = int(3*(d**0.5)*np.log(d))
        nn = max(nn, 7)  # fix for one input
        
        # default initialization
        if args == ():  
            args = ([np.copy, d], [np.tanh, nn])
        
        for neurons in args:
            # generate model parameters
            # get transformation function (= neuron type)
            assert len(neurons) > 0, "neuron type is required"
            ufunc = neurons[0]
            if (ufunc == 'lin') or (ufunc is None):
                ufunc = np.copy
            elif ufunc == 'tanh':
                ufunc = np.tanh

            # set number of neurons of that type
            if len(neurons) >= 2:
                count = neurons[1]
            elif ufunc is np.copy:
                count = d  # for linear neurons
            else:
                count = nn / (len(args)-1)
            count = max(count, 1)

            # set projection matrix and bias if available
            if len(neurons) >= 3:
                W = neurons[2]
            else:
                W = np.random.randn(d, count) / (d**0.5)
            
            if len(neurons) >= 4:
                bias = neurons[3]
            else:
                bias = np.random.rand(1,count)
           
            W = np.vstack((W, bias))
                
            # update model parameters
            ufunc = [ufunc] * count
            if self.ufunc is None:
                self.ufunc = ufunc
            else:
                self.ufunc.extend(ufunc)
            
            if self.W is None:
                self.W = W
            else:
                self.W = np.hstack((self.W, W))
        

    def train(self, X, T):
        """Wrapper for training an ELM.
        
        :param X: Training inputs
        :type X: 2-d matrix
        :param T: Training targets
        :type T: 1-d or 2-d matrix
            
        Trains the ELM model with input features and corresponding desired outputs.
        Number of inputs and outputs must be the same.
        
        If no neurons are added to the model, adds linear and tanh neurons automatically.
        """        

        X,T = self.preprocess(X,T)
        if self.inputs is None:
            self.inputs = X.shape[1] - 1  # remove bias
        if self.targets is None:
            self.targets = T.shape[1]
        
        assert self.inputs == X.shape[1] - 1, "incorrect dimensionality of inputs"
        assert self.targets == T.shape[1], "incorrect dimensionality of outputs"
        if self.W is None:
            self.init(self.inputs)

        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
            
        self.B = lstsq(H,T)[0]


    def predict(self, X):
        """Get predictions using a trained or loaded ELM model.
        
        :param X: input data
        :rtype: predictions Th
        """
        
        assert self.B is not None, "train this model first"
        X = self.preprocess(X)

        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
            
        Th = H.dot(self.B)    
        return Th
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        