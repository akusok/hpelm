# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:54:55 2014

@author: akusok
"""
import numpy as np
from tables import openFile



def encode(data, cdict):
    """Encode 1-dim classes into binary coding.
    """
    try:
        if isinstance(data, np.ndarray) and (len(data.shape)==2):
            data = data.ravel()
        data = np.vstack([cdict[cls] for cls in data])
    except:
        raise IOError("Test targets cannot have classes not presented in training")
    return data


def decode(data, cdict):
    """Transform binary coding into original 1-dim classes.
    """    
    un_cdict = {np.argmax(v): k for k,v in cdict.items()}  # invert dictionary
    return [un_cdict[i] for i in np.argmax(data, 1)]  


def batchX(X, batch=10000, delimiter=" "):
    """Iterates over data X from whatever source.
    """
    if isinstance(X, basestring) and (X[-3:] == ".h5"):  # read partially from HDF5
        h5 = openFile(X)
        for node in h5.walk_nodes(): pass  # find a node with whatever name
        N = node.shape[1]  # HDF5 files are transposed, for Matlab compatibility        
        nb = N/batch
        if N > nb*batch: nb += 1  # add last incomplete step
        for b in xrange(nb):
            start = b*batch
            step = min(batch, N-start)
            X1 = node[:, start : start+step].T
            X1 = np.hstack((X1, np.ones((step,1), dtype=X.dtype)))
            yield X1
        h5.close()  # closing file

    else:  # load whole X into memory
        # load text file
        if isinstance(X, basestring):  # any other file - must be .txt (compressed) file
            if X[-3:] in ["txt",".gz","bz2"]:
                X = np.loadtxt(X, delimiter=delimiter)
            elif X[-3:] in ['npy']:
                X = np.load(X)
            else:
                raise IOError("Input file X should be text (*.txt), "+
                              "a compressed text (*.gz/*.bz2), an HDF5 file "+
                              "(*.h5), or Numpy binary (*.npy)")
        if not isinstance(X, np.ndarray): X = np.array(X)
        if len(X.shape) == 1: X = X.reshape(-1,1)  # add second dimension
        
        # return data
        N = X.shape[0]
        nb = N/batch
        if N > nb*batch: nb += 1  # add last incomplete step
        for b in xrange(nb):
            start = b*batch
            step = min(batch, N-start)
            X1 = X[start : start+step]
            X1 = np.hstack((X1, np.ones((step,1), dtype=X.dtype)))
            yield X1  


def batchT(T, batch=10000, delimiter=" ", c_dict=None):
    """Iterates over targets T with correct transformation.
    
    :param C_dict: - dictionary of classes for single-class classification,
                     implies the classification task
    """

    if isinstance(T, basestring) and (T[-3:] == ".h5"):  # read partially from HDF5
        h5 = openFile(T)
        for node in h5.walk_nodes(): pass  # find a node with whatever name
        N = node.shape[1]  # HDF5 files are transposed, for Matlab compatibility        
        nb = N/batch
        if N > nb*batch: nb += 1  # add last incomplete step
        for b in xrange(nb):
            start = b*batch
            step = min(batch, N-start)
            T1 = node[:, start : start+step].T
            if c_dict is not None:
                T1 = encode(T1, c_dict)
            yield T1
        h5.close()  # closing file

    else:  # load whole T into memory
        # load text file
        if isinstance(T, basestring):  # any other file - must be .txt (compressed) file
            if T[-3:] in ["txt",".gz","bz2"]:
                T = np.loadtxt(T, delimiter=delimiter)
            elif T[-3:] in ['npy']:
                T = np.load(T)
            else:
                raise IOError("Targets file T should be text (*.txt), "+
                              "a compressed text (*.gz/*.bz2), an HDF5 file "+
                              "(*.h5), or Numpy binary (*.npy)")

        # checks for non-classification targets
        if c_dict is None:
            if not isinstance(T, np.ndarray): T = np.array(T)
            if len(T.shape) == 1: T = T.reshape(-1,1)  # add second dimension
        
        # return data
        N = len(T)
        nb = N/batch
        if N > nb*batch: nb += 1  # add last incomplete step
        for b in xrange(nb):
            start = b*batch
            step = min(batch, N-start)
            T1 = T[start : start+step]
            if c_dict is not None:
                T1 = encode(T1, c_dict)
            yield T1  


def meanstdX(X, batch=10000, delimiter=" "):
    """Computes mean and standard deviation of X.
    """
    if isinstance(X, basestring) and (X[-3:] == ".h5"):  # read partially from HDF5
        h5 = openFile(X)
        for node in h5.walk_nodes(): pass  # find a node with whatever name
        N = node.shape[1]  # HDF5 files are transposed, for Matlab compatibility        
        d = node.shape[0]
        nb = N/batch
        if N > nb*batch: nb += 1  # add last incomplete step

        E_x = np.zeros((d,), dtype=np.float64)
        E_x2 = np.zeros((d,), dtype=np.float64)
        for b in xrange(nb):
            start = b*batch
            step = min(batch, N-start)
            X1 = node[:, start : start+step].astype(np.float).T
            E_x += np.mean(X1,0) * (1.0*step/N)    
            E_x2 += np.mean(X1**2,0) * (1.0*step/N)    

        meanX = E_x
        E2_x = E_x**2
        stdX = (E_x2 - E2_x)**0.5            
        h5.close()  # closing file

    else:  # load whole X into memory
        # load text file
        if isinstance(X, basestring):  # any other file - must be .txt (compressed) file
            if X[-3:] in ["txt",".gz","bz2"]:
                X = np.loadtxt(X, delimiter=delimiter)
            elif X[-3:] in ['npy']:
                X = np.load(X)
            else:
                raise IOError("Input file X should be text (*.txt), "+
                              "a compressed text (*.gz/*.bz2), an HDF5 file "+
                              "(*.h5), or Numpy binary (*.npy)")
        if not isinstance(X, np.ndarray): X = np.array(X)
        if len(X.shape) == 1: X = X.reshape(-1,1)  # add second dimension

        meanX = X.mean(0)
        stdX = X.std(0)

    # fix for constant input features, prevents division by zero
    stdX[stdX == 0] = 1
    return meanX, stdX


def c_dictT(T, batch=10000):
    """Creates dictionary of classes from any targets.
    """

    if isinstance(T, basestring) and (T[-3:] == ".h5"):  # read partially from HDF5
        h5 = openFile(T)
        for node in h5.walk_nodes(): pass  # find a node with whatever name
        assert node.shape[0] == 1, "Classification targets must have only one feature"
        N = node.shape[1]  # HDF5 files are transposed, for Matlab compatibility        
        nb = N/batch
        if N > nb*batch: nb += 1  # add last incomplete step
        c_set = set([])
        for b in xrange(nb):
            start = b*batch
            step = min(batch, N-start)
            T1 = node[:, start : start+step].ravel()
            c_set = c_set.union(set(T1))            
        h5.close()  # closing file

    else:  # load whole T into memory
        # load text file
        if isinstance(T, basestring):  # any other file - must be .txt (compressed) file
            if T[-3:] in ["txt",".gz","bz2"]:
                with open(T) as f: T = f.readlines()
            elif T[-3:] in ['npy']:
                T = np.load(T)
            else:
                raise IOError("Targets file T should be text (*.txt), "+
                              "a compressed text (*.gz/*.bz2), an HDF5 file "+
                              "(*.h5), or Numpy binary (*.npy)")

        if isinstance(T, np.ndarray):
            assert (len(T.shape) == 1) or (T.shape[1] == 1), "Classification targets must have only 1 feature"
            if len(T.shape) == 2: T = T.ravel()  # make targets 1-dimensional
            
        c_set = set(T)

    classes = list(c_set)
    C = len(classes)
    temp = np.eye(C)
    C_dict = {classes[i] : temp[i] for i in xrange(C)}

    return C_dict

































