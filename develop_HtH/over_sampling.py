# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 05:16:53 2015

@author: Anton
"""

import numpy as np
import numexpr as ne
import os
from matplotlib import pyplot as plt


class elm(object):

    def __init__(self):
        self.sigm = lambda a: ne.evaluate("1/(1+exp(-a))")


    def train(self, X, T, nn, idx):
        n,d = X.shape
        np.random.seed()
        W = np.random.randn(d, nn) * 3**0.5
        bias = np.random.randn(nn)
        self.W = W
        self.bias = bias
        
        H = np.dot(X,W) + bias  # random projection
        H = self.sigm(H)  # non-linear transformation
        #self.B = np.dot(np.linalg.pinv(H), T)  # linear regression    

        ##################################################


        H1 = H[T[:,0]==1]
        T1 = T[T[:,0]==1]
        HH1 = np.dot(H1.T, H1) + 1E-6*np.eye(H1.shape[1])
        HT1 = np.dot(H1.T, T1)    

        #H2 = H[T[:,1]==1]
        #T2 = T[T[:,1]==1]
        H2 = H[idx]
        T2 = T[idx]
        HH2 = np.dot(H2.T, H2) + 1E-6*np.eye(H2.shape[1])
        HT2 = np.dot(H2.T, T2)    
        
        """
        ### RESAMPLING
        Nsampl = H2.shape[0]
        s = (np.std(X[T[:,0]==1], axis=0) + np.std(X[idx], axis=0) + np.std(X[T[:,2]==1], axis=0)) / 3
        Nres = (np.sum(T[:,0]==1) + np.sum(T[:,2]==1)) / 2
        Xres = np.zeros((Nres, d))
        for i in xrange(Nres):
            sampl = X[idx][np.random.randint(Nsampl)]
            Xres[i] = sampl + np.random.randn(1,d)*s
        H2res = np.dot(Xres,W) + bias
        H2res = self.sigm(H2res)
        T2res = np.tile(T2[0], (Nres,1))
        H2 = np.vstack((H2, H2res))
        T2 = np.vstack((T2, T2res))
        HH2 = np.dot(H2.T, H2) + 1E-6*np.eye(H2.shape[1])
        HT2 = np.dot(H2.T, T2)    
        #"""
        Xres=np.zeros((3,2))


        H3 = H[T[:,2]==1]
        T3 = T[T[:,2]==1]
        HH3 = np.dot(H3.T, H3) + 1E-6*np.eye(H3.shape[1])
        HT3 = np.dot(H3.T, T3)    

        N = n*1.0
        k = np.array([1, 1, 1], dtype=np.float64)
        k = np.array([N/H1.shape[0], N/H2.shape[0], N/H3.shape[0]]).astype(np.float64)
        k = k / np.sum(k)
        HH = k[0]*HH1 + k[1]*HH2 + k[2]*HH3
        HT = k[0]*HT1 + k[1]*HT2 + k[2]*HT3
        
        
        ##################################################
        
        #HH = H.T.dot(H) + 1E-6*np.eye(H.shape[1])
        #HT = H.T.dot(T)    
        self.B = np.linalg.lstsq(HH, HT)[0]
        return Xres
        
        


    def run(self, X):
        H = np.dot(X, self.W) + self.bias
        H = self.sigm(H)
        return np.dot(H, self.B)



def code(nn):
    folder = 'Classification-Iris'
    folder = os.path.join(os.path.dirname(__file__), "../datasets", folder)
    acc = np.zeros((10,))
    for i in range(10):  # 10-fold cross-validation
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ytr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Yts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        Xtr = Xtr[:,2:]
        Xts = Xts[:,2:]
        Ytr = np.hstack((Ytr[:,:2], Ytr[:,3][:,None]))
        Yts = np.hstack((Yts[:,:2], Yts[:,3][:,None]))

        e = elm()
        e.train(Xtr, Ytr, nn)        
        Yh = e.run(Xts)

        # evaluate classification results
        Yts = np.argmax(Yts, 1)
        Yh = np.argmax(Yh, 1)
        acc[i] = float(np.sum(Yh == Yts)) / Yts.shape[0]
    return acc


def code_show(nn, i):
    folder = 'Classification-Iris'
    folder = os.path.join(os.path.dirname(__file__), "../datasets", folder)
    acc = np.zeros((10,))
    # get file names
    Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
    Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
    Ytr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
    Yts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
    # train ELM
    #Xtr = Xtr[:,2:]
    #Xts = Xts[:,2:]
    i1 = 0
    i2 = 3
    Xtr = np.vstack((Xtr[:,i1], Xtr[:,i2])).T
    Xts = np.vstack((Xts[:,i1], Xts[:,i2])).T

    Ytr = np.hstack((Ytr[:,:2], Ytr[:,3][:,None]))
    Yts = np.hstack((Yts[:,:2], Yts[:,3][:,None]))

    #print '1'
    #print Xtr[Ytr[:,0]==1].std(axis=0)**2
    #print Xtr[Ytr[:,1]==1].std(axis=0)**2


    # prepare mesh grid
    x = np.arange(-2.5, 2.5, 0.1)
    y = np.arange(-2.5, 2.5, 0.1)
    n = x.shape[0]
    xm, ym = np.meshgrid(x, y)
    D = np.vstack((xm.ravel(), ym.ravel())).T

    # reduce number of samples of one class
    N2 = 2
    idx = np.where(Ytr[:,1]==1)[0]
    np.random.shuffle(idx)
    idx = idx[:N2]
    idx = np.array([21, 76])
    print idx
    
    # average over many runs
    Z = []
    acc = []
    for _ in range(1):
    
        e = elm()
        Xres = e.train(Xtr, Ytr, nn, idx)
    
        # evaluate classification results
        Yh = e.run(Xts)
        Yts1 = np.argmax(Yts, 1)
        Yh1 = np.argmax(Yh, 1)
        acc.append(float(np.sum(Yh1 == Yts1)) / Yts1.shape[0])
    
        # show plot
        Z.append(e.run(D))
        
    Z = np.array(Z).mean(0)
    print "acc:", np.mean(acc)

    I = Z.copy()
    Z[Z<0] = 0
    I[I<0] = 0
    I = I.reshape(n,n,3)
    I = I / I.max(axis=2)[:,:,None]

    I = I[::-1, :, :]
        
    
    plt.imshow(I, extent=[-2.5, 2.5, -2.5, 2.5])
    #cl = 1
    #plt.contour(xm, ym, Z[:,cl].reshape(n,n), colors='k')
    #plt.contourf(xm, ym, Z[:,cl].reshape(n,n), cmap=plt.cm.bone)
    #plt.scatter(D[:,0], D[:,1], s=Z[:,0])
    
    #Xa = np.vstack((Xtr, Xts))
    #Ya = np.vstack((Ytr, Yts))
    #plt.scatter(Xa[:,0], Xa[:,1], c=Ya)
    #plt.scatter(Xtr[:,0], Xtr[:,1], c=Ytr)
    plt.scatter(Xtr[Ytr[:,0]==1,0], Xtr[Ytr[:,0]==1,1], c=Ytr[Ytr[:,0]==1])
    plt.scatter(Xtr[idx,0], Xtr[idx,1], c=Ytr[idx])
    plt.scatter(Xres[:,0], Xres[:,1], c=Ytr[idx[0]])
    plt.scatter(Xtr[Ytr[:,2]==1,0], Xtr[Ytr[:,2]==1,1], c=Ytr[Ytr[:,2]==1])
    plt.show()

    



if __name__ == "__main__":
    nn = 7
    #m = []
    #for _ in range(100):
    #    acc = code(nn)
    #    m.append(np.mean(acc))
    #m = np.array(m)*100
    #print "%.1f+-%.1f" % (np.mean(m), np.std(m))

    code_show(nn, 0)

























