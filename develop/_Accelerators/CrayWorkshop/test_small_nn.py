'''
Created on May 14, 2014

@author: akusoka1
'''

import numpy as np
from time import time
    
def test_small():    
    b = 2**28  # 1GB RAM
    
    A = np.random.random_sample((b,)).astype(np.float64)
    B = np.random.random_sample((b,)).astype(np.float64)
    print A.shape, B.shape
    
    for nn in [1,1,2,4,8,16,32,64,128,256,512,1024]:
        N = b/nn
        A = A.reshape(nn,N)
        B = B.reshape(N,nn)
        
        t = time()
        np.dot(A,B)
        t = time() - t
        
        print "%.1f GB RAM, %d neurons: %.2f seconds, complex %.1f" % (((nn*N*8.0) / 2**29), nn, t, (1.0*nn*N*nn)/2**28)
        
        
def test_large():    
    b = 2**28  # 1GB RAM
    
    A = np.random.random_sample((b,)).astype(np.float64)
    B = np.random.random_sample((b,)).astype(np.float64)
    print A.shape, B.shape
    
    for nn in [5000,10000]:
        N = b/nn
        #A = A.ravel()[:nn*N].reshape(nn,N)
        #B = B.ravel()[:nn*N].reshape(N,nn)
        C = np.random.rand(nn,nn)
        
        #t = time()
        #np.linalg.pinv(C)
        #t1 = time() - t
        #t = time()
        #np.linalg.svd(C)
        #t2 = time() - t
        t1=0
        t2=0
        
        B = np.random.rand(nn,10)
        t = time()
        Q,R = np.linalg.qr(C)
        P = np.dot(Q.T, B)
        np.dot(np.linalg.inv(R), P)
        t3 = time() - t
        
        print "%.1f GB RAM, %d neurons: inv %.2fs, SVD %.2fs, QR %.2fs" % (((nn*N*8.0) / 2**29), nn, t1, t2, t3)
        

if __name__ == '__main__':
    test_large()