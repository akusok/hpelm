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


class mrsr2(object):

    
    def __init__(self, X, T, norm=1):
        """Builds linear model.
        
        X is input data.
        T aer real outputs (targets).
        Y are estimated outputs.
        """
        ins = X.shape[1]
        outs = T.shape[1]

        self.norm = norm
        self.X = np.array(X, order='F')  # Fortran ordering is good for operating columns 
        self.XA = np.empty(X.shape, order='F')
        self.T = T
        self.XX = np.dot(X.T, X)        
        self.XT = np.dot(X.T, T)
        
        self.rank = []  # active inputs list
        self.nonrank = range(ins)
        self.W = np.zeros((ins, outs))  # current projection estimator
        self.Y = np.zeros(T.shape)  # current target estimator
        self.j_current = None  # currently added input dimension
        
        
    #@profile
    def new_input(self):
        """Adds one input to a current model using a variable length step.
        """
        
        # first step
        if len(self.rank) == 0:
            c_max = -1
            for j in self.nonrank:
                c_kj = np.linalg.norm(np.dot(self.T.T, self.X[:,j]), self.norm)
                if c_kj > c_max:
                    c_max = c_kj
                    j_max = j
            self.j_current = j_max  
            # save new input
            self.rank.append(self.j_current)
            self.nonrank.remove(self.j_current)
            # swap columns
            idx = len(self.rank)-1
            self.XA[:,idx] = self.X[:,self.j_current]
            y_min = 1
        
        # last step
        elif len(self.nonrank) == 0:  
            print "last step"
            self.Y = self.Yk1
            self.W[self.rank] = self.Wk1
            y_min = 1
               
        # intermediate step        
        else:  
            Yk2 = (self.Yk1 - self.Y).T
            T2 = (self.T - self.Y).T
            c_max =  np.linalg.norm(np.dot(T2, self.X[:,self.j_current]), self.norm)
            
            #fun = lambda y,x_new: (1-y)*c_max - np.linalg.norm(T2.dot(x_new) - y*Yk2.dot(x_new))
            fun_p = lambda y,p1,p2: (1-y)*c_max - np.linalg.norm(p1 - y*p2)  # super fast parametrized function
    
            # find optimal step (minimum over possible additional inputs) 
            y_min = 1  # upper interval
            for j_new in self.nonrank:
                x_new = self.X[:,j_new]
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
            self.j_current = j_min

            # add new input into model
            self.rank.append(self.j_current)
            self.nonrank.remove(self.j_current)
            # add new input to X matrix
            idx = len(self.rank)-1
            self.XA[:,idx] = self.X[:,self.j_current]

        # post-update ELM estimation with current set of inputs, with LU-ELM
        XtX = self.XX[self.rank,:][:,self.rank]
        XtT = self.XT[self.rank,:]
        LU, piv = lu_factor(XtX)#, overwrite_a=True)
        self.Wk1 = lu_solve((LU, piv), XtT)#, overwrite_b=True)
        X = self.XA[:,:len(self.rank)]  # replace fancy indexing with simple one
        self.Yk1 = np.dot(X, self.Wk1)  

        if len(self.rank) > 1:
            # perform variable length step
            self.Y = (1-y_min)*self.Y + y_min*self.Yk1
            self.W = (1-y_min)*self.W     
            self.W[self.rank] += y_min*self.Wk1   

        return y_min
       

    def get_ols_solution(self):        
        """Return OLS solution for a current model.
        
        Current estimates of Y and W are useful in the model,
        but are very bad for prediction.
        """
        X = self.XA[:,:len(self.rank)]
        XtX = self.XX[self.rank,:][:,self.rank]
        XtT = self.XT[self.rank,:]
        LU, piv = lu_factor(XtX, overwrite_a=True)
        Wk1 = lu_solve((LU, piv), XtT, overwrite_b=True)
        Yk1 = np.dot(X, Wk1)            
        return Wk1, Yk1
        























































