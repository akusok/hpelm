# -*- coding: utf-8 -*-
"""
Better MRSR implementation according to:
'Common subset selection of inputs in multiresponse regression" by
Timo Similä and Jarkko Tikka, 
International Joint Conference on Neural Networks 2006

Linear complexity with the number of targets.

Created on Sun Jan 26 13:48:54 2014
@author: Anton Akusok
"""

import numpy as np
from scipy import optimize
from scipy.linalg import lu_factor, lu_solve


def mrsr2(X, T, kmax, norm=2):

    # initializing
    ins = X.shape[1]
    outs = T.shape[1]
    X = np.array(X, order='F')  # Fortran ordering is good for operating columns 
    XA = np.empty(X.shape, order='F')
    XX = np.dot(X.T, X)        
    XT = np.dot(X.T, T)
    rank = []  # active inputs list
    nonrank = range(ins)
    W = np.zeros((ins, outs))  # current projection estimator
    Y = np.zeros(T.shape)  # current target estimator
    Yk1 = np.zeros(T.shape)
    Wk1 = np.zeros((1, outs))
    j_current = None  # currently added input dimension
    kmax = min(kmax, X.shape[0])    
    
    # get ranking
    for _ in xrange(kmax):
        
        # first step
        if len(rank) == 0:
            c_max = -1
            for j in nonrank:
                c_kj = np.linalg.norm(np.dot(T.T, X[:,j]), norm)
                if c_kj > c_max:
                    c_max = c_kj
                    j_max = j
            j_current = j_max  
            # save new input
            rank.append(j_current)
            nonrank.remove(j_current)
            # swap columns
            idx = len(rank)-1
            XA[:,idx] = X[:, j_current]
            y_min = 1
        
        # last step
        elif len(nonrank) == 0:  
            Y = Yk1
            W[rank] = Wk1
            y_min = 1

        # intermediate step        
        else:  
            Yk2 = (Yk1 - Y).T
            T2 = (T - Y).T
            
            X9 = X[:, j_current]
            c_max = np.linalg.norm(np.dot(T2, X9), norm)
            #c_max = np.linalg.norm(np.dot(T2, X[:, j_current]), norm)
            
            #fun = lambda y,x_new: (1-y)*c_max - np.linalg.norm(T2.dot(x_new) - y*Yk2.dot(x_new))
            fun_p = lambda y,p1,p2: (1-y)*c_max - np.linalg.norm(p1 - y*p2)  # super fast parametrized function
    
            # find optimal step (minimum over possible additional inputs) 
            y_min = 1  # upper interval
            for j_new in nonrank:
                x_new = X[:,j_new]
                # pre-calculate constant parts of the optimization function for the given x_new
                p1 = T2.dot(x_new)
                p2 = Yk2.dot(x_new)
                if (1-y_min)*c_max < np.linalg.norm(p1 - y_min*p2):  # skip optimization if min(fun) > y_min
                    try:
                        zero = 1E-15  # finding a value greater than zero
                        y_kj = optimize.brentq(fun_p, zero, y_min, xtol=1E-6, args=(p1,p2))
                        y_min = y_kj
                        j_min = j_new
                    except ValueError:  
                        # ValueError: f(a) and f(b) must have different signs
                        # here f(a) < 0 and f(b) < 0; does not fit our purposes anyway 
                        # ignoring this case
                        pass                  
                    
            if y_min == 1:  # if no suitable solution was found
                j_min = j_new
            j_current = j_min

            # add new input into model
            rank.append(j_current)
            nonrank.remove(j_current)
            # add new input to X matrix
            idx = len(rank)-1
            XA[:,idx] = X[:,j_current]

        # post-update ELM estimation with current set of inputs, with LU-ELM
        XtX = XX[rank,:][:,rank]
        XtT = XT[rank,:]
        LU, piv = lu_factor(XtX)#, overwrite_a=True)
        Wk1 = lu_solve((LU, piv), XtT)#, overwrite_b=True)
        X1 = XA[:,:len(rank)]  # replace fancy indexing with simple one
        Yk1 = np.dot(X1, Wk1)  

        if len(rank) > 1:
            # perform variable length step
            Y = (1-y_min)*Y + y_min*Yk1
            W = (1-y_min)*W     
            W[rank] += y_min*Wk1   
               
    # done, return ranking
    return rank

        























































