# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
from numpy.linalg import lstsq
from tables import openFile, Atom



class ELM(object):
    """Non-parallel Extreme Learning Machine.
    """

    ###########################################################################
    ### helper methods ###


    @staticmethod
    def h5write(filename, varname, data):
        """Writes one data matrix to HDF5 file.
        
        Similar to Matlab function.
        """
        assert isinstance(filename, basestring), "file name must be a string"
        assert isinstance(varname, basestring), "variable name must be a string"
        assert isinstance(data, np.ndarray), "data must be a Numpy array"
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        
        # remove leading "/" from variable name
        if varname[0] == "/":
            varname = varname[1:]

        try:
            h5 = openFile(filename, "w")
            a = Atom.from_dtype(data.dtype)
            h5.create_array(h5.root, varname, data.T, atom=a)  # transpose for Matlab compatibility
            h5.flush()
        finally:
            h5.close()
        
        
    @staticmethod
    def h5read(filename):
        """Reads one data matrix from HDF5 file, variable name does not matter.
        
        Similar to Matlab function.
        """
        h5 = openFile(filename)
        for node in h5.walk_nodes():  # find the last node with whatever name
            pass
        M = node[:].T  # transpose for Matlab compatibility
        h5.close()
        return M    


    ###########################################################################
    ### main methods ###


    def __init__(self, *args):
        """Create ELM of desired kind.
        
        :param regression: type of ELM task, can be regression, classification or timeseries regression.
        :param sparse: set to create an ELM with sparse projection matrix.
        """
        self.classification = False
        self.multiclass = False
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
        self.N = None
        self.W = None
        self.ufunc = None
        self.B = None
        self.Xmean = 0
        self.Xstd = 1
        self.Tmean = 0
        self.Tstd = 1
        self.Tbase = None  # 0-based or 1-based classes
        self.C = None


    def data_loader(self, X, T=None, training=False, batch=None, delimiter=None):
        """Returns pre-processed data or data iterator.
        
        :param X: input data location (variable or file name)
        :param T: targets data location (variable or file name)
        :param training: evaluates normalization parameters on training
        :param batch: if batch size is given, returns data iterator
        :param delimiter: can specify delimiter to open text files
        """        
        
        ### evaluate normalization parameters, requires both X and T ###
        if training:
            assert T is not None, "targets (outputs) must be specified for training"

            ### process input data X ###
            if isinstance(X, basestring) and (X[-3:] == ".h5"):
                # HDF5
                h5 = openFile(X)
                for node in h5.walk_nodes(): pass  # find a node with whatever name
                assert len(node.shape) == 2, "HDF5 data must be 2-dimensional"
                Xshape = node.shape[::-1]  # take into account Matlab transpose
                self.N = Xshape[0]  
                self.inputs = Xshape[1]
                
                # batch compute mean and std**2 = mean(x**2) - mean(x)**2
                N = self.N
                if batch is None: 
                    b = N
                else:
                    b = batch
                E_x = np.zeros((Xshape[1],), dtype=np.float64)
                E_x2 = np.zeros((Xshape[1],), dtype=np.float64)
                
                for i in xrange(N/b + 1):
                    k = min(b, N-i*b)
                    if k == 0: break
                    xb = node[:, i*b : i*b + k].astype(np.float)
                    E_x += np.mean(xb,1) * (1.0*k/N)    
                    E_x2 += np.mean(xb**2,1) * (1.0*k/N)    
                        
                self.Xmean = E_x
                E2_x = E_x**2
                self.Xstd = (E_x2 - E2_x)**0.5
                h5.close()
            else:
                # load .txt (compressed) file as Numpy array 
                if isinstance(X, basestring):
                    assert X[-3:] in ["txt",".gz","bz2"], "input file X should be *.txt or compressed *.gz/*.bz2"
                    if delimiter is None: delimiter = " "
                    X = np.loadtxt(X, delimiter=delimiter)
                    
                # passing data as Numpy array
                if not isinstance(X, np.ndarray): X = np.array(X)
                if len(X.shape) == 1: X = X.reshape(-1,1)
                self.N = X.shape[0]
                self.Xmean = X.mean(0)
                self.Xstd = X.std(0)
                self.inputs = X.shape[1]

            # fix for constant input features
            self.Xstd[self.Xstd == 0] = 1
            

            ### process input targets T ###
            if isinstance(T, basestring) and (T[-3:] == ".h5"):
                # HDF5
                h5 = openFile(T)
                for node in h5.walk_nodes(): pass  # find a node with whatever name
                assert len(node.shape) == 2, "HDF5 targets must be 2-dimensional"
                Tshape = node.shape[::-1]  # take into account Matlab transpose
                assert self.N == Tshape[0], "inputs and targets must have the same amount of samples"
                self.targets = Tshape[1]
                
                N = self.N
                if batch is None: 
                    b = N
                else:
                    b = batch
                    
                if not self.classification:                
                    # batch compute mean and std**2 = mean(x**2) - mean(x)**2
                    E_t = np.zeros((Tshape[1],), dtype=np.float64)
                    E_t2 = np.zeros((Tshape[1],), dtype=np.float64)
                    
                    for i in xrange(N/b + 1):
                        k = min(b, N-i*b)
                        if k == 0: break
                        tb = node[:, i*b : i*b + k].astype(np.float)
                        E_t += np.mean(tb,1) * (1.0*k/N)    
                        E_t2 += np.mean(tb**2,1) * (1.0*k/N)                                
                    self.Tmean = E_t
                    E2_t = E_t**2
                    self.Tstd = (E_t2 - E2_t)**0.5

                elif Tshape[1] == 1:  # in case of integer classes in classification
                    # batch check classification targets
                    self.C = 0
                    tmin = 2
                    tmax = -1
                    for i in xrange(N/b + 1):
                        k = min(b, N-i*b)
                        if k == 0: break
                        tb = node[i*b : i*b + k]
                        assert np.allclose(np.array(tb,np.int64), tb), "classification targets must be integers"
                        tmin = min(tmin, np.min(tb))                        
                        tmax = max(tmax, np.max(tb))
                    assert tmin in [0,1], "classes must start from 0 or 1"
                    self.Tbase = tmin
                    self.C = tmax + (1-tmin)
                    self.targets = self.C
                else:
                    self.multiclass = True  # assume multiclass classification
                h5.close()                
            else:
                # load .txt (compressed) file as Numpy array 
                if isinstance(T, basestring):
                    assert T[-3:] in ["txt",".gz","bz2"], "input targets T should be *.txt or compressed *.gz/*.bz2"
                    if delimiter is None: delimiter = " "
                    T = np.loadtxt(T, delimiter=delimiter)
                    
                # passing data as Numpy array
                if not isinstance(T, np.ndarray): T = np.array(T)
                assert self.N == T.shape[0], "inputs and targets must have the same amount of samples"
                if len(T.shape) == 1: T = T.reshape(-1,1)                
                self.targets = T.shape[1]
                
                # normalization only for regression
                if not self.classification:  
                    self.Tmean = T.mean(0)
                    self.Tstd = T.std(0)
                elif T.shape[1] == 1:  # in case of integer classes in classification
                    assert np.allclose(np.array(T,np.int64), T), "classification targets must be integers"
                    assert np.min(T) in [0,1], "classes must start from 0 or 1"
                    self.Tbase = np.min(T)
                    self.C = T.max() + (1-self.Tbase)
                    self.targets = self.C
                else:
                    self.multiclass = True
                
            
        ### preparing data as one chunk ###
        if batch is None:            
            if isinstance(X, basestring) and (X[-3:] == ".h5"):
                # HDF5
                h5 = openFile(X)
                for node in h5.walk_nodes(): pass  # find a node with whatever name
                X = node[:].T
                h5.close()
            elif isinstance(X, basestring):  # any other file - must be .txt (compressed) file
                assert X[-3:] in ["txt",".gz","bz2"], "input file X should be *.txt or compressed *.gz/*.bz2"
                if delimiter is None: delimiter = " "
                X = np.loadtxt(X, delimiter=delimiter)
            if not isinstance(X, np.ndarray): X = np.array(X)
            if len(X.shape) == 1: X = X.reshape(-1,1)

            X = (X - self.Xmean) / self.Xstd
            self.N = X.shape[0]
            
            X = np.hstack((X, np.ones(self.N).reshape((-1,1))))
           
            if T is not None:
                if isinstance(T, basestring) and (T[-3:] == ".h5"):
                    # HDF5
                    h5 = openFile(T)
                    for node in h5.walk_nodes(): pass  # find a node with whatever name
                    T = node[:].T
                    h5.close()
                elif isinstance(T, basestring):
                    assert T[-3:] in ["txt",".gz","bz2"], "input targets T should be *.txt or compressed *.gz/*.bz2"
                    if delimiter is None: delimiter = " "
                    T = np.loadtxt(T, delimiter=delimiter)
                if not isinstance(T, np.ndarray): T = np.array(T)

                assert self.N == T.shape[0], "inputs and targets must have the same amount of samples"
                if len(T.shape) == 1: T = T.reshape(-1,1)

                if not self.classification:
                    T = (T - self.Tmean) / self.Tstd
                elif T.shape[1] == 1:
                    # preparing classification
                    assert self.C is not None, "train model first to initialize classes" 
                    assert self.Tbase is not None, "train model first to initialize classes" 
                    T2 = np.zeros((self.N, self.C), dtype=np.int8)
                    T2[xrange(self.N), (T-self.Tbase).ravel()] = 1
                    T = T2                  
                return X,T
            else:
                return X
            
        ### prepare a data generator with given batch size ###
        else:
            raise NotImplemented


    def add_neurons(self, inputs, *args):
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
            if ufunc is np.copy:
                count = d  # for linear neurons
            elif len(neurons) >= 2:
                count = neurons[1]
            else:
                count = nn / (len(args)-1)
            count = max(count, 1)

            # set projection matrix and bias if available
            if ufunc is np.copy:
                W = np.eye(d)
            elif len(neurons) >= 3:
                W = neurons[2]
            else:
                W = np.random.randn(d, count) / (d**0.5)
            
            if len(neurons) >= 4:
                bias = neurons[3]
            else:
                bias = np.random.rand(1,count)
           
            W = np.vstack((W, bias))

            # create list of transformation functions
            if not hasattr(ufunc, '__iter__'):
                ufunc = [ufunc] * count
            assert hasattr(ufunc[0], '__call__'), "provide function, a list of functions, or 'lin'/'tanh'/None"
                
            # update model parameters
            if self.ufunc is None:
                self.ufunc = ufunc
            else:
                self.ufunc.extend(ufunc)
            
            if self.W is None:
                self.W = W
            else:
                self.W = np.hstack((self.W, W))


    def train(self, X, T, *args):
        """Wrapper for training an ELM.
        
        :param X: Training inputs
        :type X: 2-d matrix
        :param T: Training targets
        :type T: 1-d or 2-d matrix
            
        Trains the ELM model with input features and corresponding desired outputs.
        Number of inputs and outputs must be the same.
        
        If no neurons are added to the model, adds linear and tanh neurons automatically.
        """        
        
        X,T = self.data_loader(X,T,training=True)
        
        assert self.inputs == X.shape[1] - 1, "incorrect dimensionality of inputs"
        assert self.targets == T.shape[1], "incorrect dimensionality of outputs"
        if self.W is None: self.add_neurons(self.inputs, *args)

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
        X = self.data_loader(X)
        assert self.inputs == X.shape[1] - 1, "incorrect dimensionality of inputs"

        H = np.dot(X,self.W)
        for i in xrange(H.shape[1]):
            H[:,i] = self.ufunc[i](H[:,i])
            
        Th = H.dot(self.B)  
        if self.classification:
            if self.multiclass:
                Th = np.array(Th > 0.5, dtype=np.int)
            else:
                Th = np.argmax(Th,1) + self.Tbase
        else:
            # de-normalize to original values
            Th = (Th * self.Tstd) + self.Tmean  
        return Th
        

    def loo_press(self, X, Y):
        """PRESS (Predictive REsidual Summ of Squares) error.
        
        Trick is to never calculate full HPH' matrix.
        """
        X,Y = self.data_loader(X,Y)
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        