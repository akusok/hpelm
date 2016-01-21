# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:40:03 2015

@author: Anton
"""
import numpy as np
from numbapro import cuda
from numbapro.cudalib import cublas
from timeit import default_timer as timer
from time import time



def run():
    L = 10000
    b = 10000
    k = 3
    
    cuda.select_device(0)    
    blas = cublas.Blas()

    from scipy.linalg.blas import ssyrk as syrk
    p = np.float32
    H = np.random.rand(b, L).astype(p)
    HH = np.zeros((L,L), dtype=p)
    HH1 = np.zeros((L, L), dtype=p)
    HH2 = np.zeros((L, L), dtype=p)
    Hf = H.copy(order='F')
    
    for _ in xrange(k):
        t = time()
        blas.syrk('L', 'T', L, b, 1, Hf, 1, HH)
        print 'numba syrk', time()-t
    cuda.close()

    for _ in xrange(k):
        t = time()
        HH1 += np.dot(H.T, H)
        print 'np dot', time()-t

    for _ in xrange(k):
        t = time()
        HH2 =syrk(1, H.T, 1, HH2, trans=0)#, overwrite_c=1)
        print 'np dsyrk', time()-t
    
    HH2 = HH2 + np.triu(HH2, k=1).T
    HH = HH + np.triu(HH, k=1).T
    print "numba syrk"
    print HH[:2, :4]
    print "np dot"
    print HH1[:2, :4]
    print "np syrk"
    print HH2[:2, :4]
    
    print np.allclose(HH, HH1)
    print np.allclose(HH, HH2)




def runtime():
    L = 10000
    from scipy.linalg.blas import dsyrk

    datas = np.zeros((200, 5))
    blas = cublas.Blas()

    i = 0
    for b in [10000]: #xrange(100, 20001, 100):

        H = np.random.rand(b, L)
        print
        print "b = %d" % b
    
        start = timer()
        HH0 = np.dot(H.T, H)
        t0 = timer() - start
        print "np dot %.4f (1)" % t0
    
        Hf = np.array(H, order='F')
        HH = np.zeros((L, L))
        start = timer()
        HH = dsyrk(1, Hf, 1, HH, trans=1)
        t1 = timer() - start
        print "np syrk/f %.4f (%.1f)" % (t1, t0/t1)
        HH = HH + np.triu(HH, k=1).T
        assert np.allclose(HH, HH0)    
    
        HH = np.zeros((L, L))
        start = timer()
        HH = dsyrk(1, H.T, 1, HH, trans=0)
        t2 = timer() - start
        print "np syrk/c %.4f (%.1f)" % (t2, t0/t2)
        HH = HH + np.triu(HH, k=1).T
        assert np.allclose(HH, HH0)    
    
        HH = np.zeros((L, L))    
        Hf = np.array(H, order='F')
        start = timer()
        blas.gemm('T', 'N', L, L, b, 1.0, Hf, Hf, 1.0, HH)
        t3 = timer() - start
        print "numba gemm/f %.4f (%.1f)" % (t3, t0/t3)
        assert np.allclose(HH, HH0)    
 
        HH = np.zeros((L, L))    
        Hf = np.array(H, order='F')
        start = timer()
        blas.syrk('L', 'T', L, b, 1, Hf, 1, HH)
        t4 = timer() - start
        print "numba syrk/f %.4f (%.1f)" % (t4, t0/t4)
        HH = HH + np.triu(HH, k=1).T
        assert np.allclose(HH, HH0)    

       
        datas[i] = [t0, t1, t2, t3, t4]
        i += 1
#        break
#        np.savetxt("datas_%d.txt" % i, datas, fmt="%.5f")
        

def order():

    L = 3000
    l = L
    batch = 1200
    dim = 250
    func = {'tan': np.tanh, 'lin': lambda x: x}
    neurons = (('lin', l/3, np.random.rand(dim, l/3), np.random.rand(l/3,)),
               ('tan', l-l/3, np.random.rand(dim, l-l/3), np.random.rand(l-l/3,)))
    X = np.random.rand(batch, dim)
    
    start = timer()
    H = np.empty((batch, L), order='F')
    i = 0
    for ftype, l1, W, B in neurons:
        H[:, i:i+l1] = func[ftype](np.dot(X, W) + B)
        i += l1
    print timer()-start
        
    start = timer()
    H2 = np.hstack([func[ftype](np.dot(X, W) + B) for ftype, _, W, B in neurons])
    H2 = np.asfortranarray(H2)
    print timer()-start

    print H.flags
    print H2.flags
    print np.allclose(H, H2)


def test_HT():
    blas = cublas.Blas()
    
    L = 100
    c = 10
    batch = 120    
    
    H = np.random.rand(batch, L)
    T = np.random.rand(batch, c)

    HT = np.zeros((L, c))
    Hf = np.asfortranarray(H)
    Tf = np.asfortranarray(T)
    HTf = np.asfortranarray(HT)
    blas.gemm('T', 'N', L, c, batch, 1.0, Hf, Tf, 1.0, HTf)
    HT = HTf

    HT0 = np.dot(H.T, T)
    print np.allclose(HT, HT0)






if __name__ == "__main__":
    runtime()
    print 'done'























