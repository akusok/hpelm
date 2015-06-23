# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:21:00 2014

@author: akusok
"""

import numpy as np
from mpi4py import MPI


def divide_X(X, size):
    n = X.shape[0]
    if n%size == 0:
        batch = n / size
    else:
        batch = (n / size) + 1

    Xd = []
    for i in range(size):
        Xd.append(X[batch*i:batch*(i+1)])
    
    return Xd


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



if rank == 0:  # master node
    # distribute projection matrix
    W = np.random.rand(20,100)
    W = comm.bcast(W, root=0)
    print "%d W: "%rank, W.shape


    # distribute nonlinear transformation function
    F = []
    #F.extend([lambda x:x]*20)
    F.extend([np.tanh]*100)
    F = comm.bcast(F, root=0)


    # distribute input data
    X_input = np.random.randn(10000,20)
    Xd = divide_X(X_input, size)  
    #del X_input
    X = comm.scatter(Xd, root=0)
    print "%d X: "%rank, X.shape  


    # do computations
    Hp = X.dot(W).astype('d')        
    H = np.empty(Hp.shape)
    for i in range(len(F)):
        H[:,i] = F[i](Hp[:,i])
    HpH = Hp.T.dot(Hp)
        

    # obtain joined result        
    HH = np.empty(HpH.shape, dtype='d')
    comm.Allreduce([HpH, MPI.DOUBLE], [HH, MPI.DOUBLE], op=MPI.SUM)
    

    # check results    
    H2 = X_input.dot(W)
    H2H = H2.T.dot(H2)
    print "results are the same: ", np.allclose(HH, H2H)    
    
    print "%d done!" % rank

else:  # worker nodes
    # distribute projection matrix
    W = comm.bcast(None, root=0)
    print "%d W: "%rank, W.shape


    # distribute nonlinear transformation function
    F = comm.bcast(None, root=0)


    # distribute input data
    X = comm.scatter(None, root=0)
    print "%d X: "%rank, X.shape  
    
    # do computations
    Hp = X.dot(W).astype('d')
    HpH = Hp.T.dot(Hp)
    
    # obtain joined result    
    HH = np.empty(HpH.shape, dtype='d')
    comm.Allreduce([HpH, MPI.DOUBLE], [HH, MPI.DOUBLE], op=MPI.SUM)
    
    print "%d done!" % rank











