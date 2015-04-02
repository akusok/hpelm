# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:21:55 2015

@author: akusok
"""

import numpy as np
import sys
from elm import ELM


def tsine(n=1000):
    err = 0.1
    X = np.linspace(-1, 1, num=n)
    T = np.sin(16 * X) * X + np.random.randn(n) * err
    Topt = np.sin(16 * X) * X

    idx = np.arange(n)
    np.random.shuffle(idx)
    itr = idx[:int(n*0.7)]
    ivl = idx[int(n*0.7):]

    Xtr = X[itr]
    Ttr = T[itr]
    Xv = X[ivl]
    Tv = T[ivl]

    elm = ELM(1, 1)
    elm.add_neurons(100, "tanh")
    # elm.train(Xtr, Ttr, "V", Xv=Xv, Tv=Tv)
    # elm.train(X, T, "LOO", "OP", kmax=20)
    elm.train(X, T, "CV", "OP", k=5)
    Yts = elm.predict(X)

    from matplotlib import pyplot as plt
    plt.plot(X, Topt, '-k')
    plt.scatter(Xtr, Ttr, color='k', s=1)
    plt.plot(X, Yts, '-r')

    plt.show()

    print T[:3]
    print Yts[:3]

    MSE = np.mean((T - Yts[:, 0]) ** 2)
    print "MSE", MSE


def tiris():
    pth = "../datasets/Classification-Iris/"
    Xtr = np.load(pth+"xtrain_1.npy")
    Ttr = np.load(pth+"ytrain_1.npy")[:, (0, 1, 3)]
    Xts = np.load(pth+"xtest_1.npy")
    Tts = np.load(pth+"ytest_1.npy")[:, (0, 1, 3)]

    elm = ELM(4, 3)
    elm.add_neurons(20, "sigm")
    elm.train(Xtr, Ttr, "LOO")

    Yts = elm.predict(Xts)
    print np.sum(Yts.argmax(1) == Tts.argmax(1)), np.sum(Yts.argmax(1) != Tts.argmax(1))

    print "Iris done!"


def trandom():
    n = int(sys.argv[1])
    d = int(sys.argv[2])
    batch = int(sys.argv[3])
    nn = 3000
    nt = 10
    print "%d/%d in, %d nn, %d out" % (n,d,nn,nt)
    X = np.random.rand(n, d)
    T = np.random.rand(n, nt)
    X = (X - X.mean(0)) / X.std(0)
    
    print "start"
    elm = ELM(d, nt, accelerate="GPU", batch=batch)
    elm.add_neurons(nn, "tanh")
    elm.train(X, T)
    #Th = elm.predict(X)

    #print np.mean((T-Th)**2)
    print "done"


#tsine()
#trandom()

class my1:
    a = 3    
    def f0(self):
        b = 5
        def ff0(self, b):
            print "a", self.a
            print "b", b
        ff0(self, b)

m = my1()
m.f0()


