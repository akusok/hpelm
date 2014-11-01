# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 18:38:28 2014

@author: akusok
"""

import numpy as np
from hpelm import ELM
from time import time
import os


datasets = os.path.join(os.path.dirname(__file__), "../datasets_big/")

def run_sine():
    path = os.path.join(datasets, "sine")
    elm = ELM()
    x = os.path.join(path, "sine_x.txt")
    y = os.path.join(path, "sine_y.txt")

    x = np.loadtxt(x)
    y = np.loadtxt(y)
    #y = (y - y.mean()) / y.std()

    t = time()
    elm.train(x,y, (None,2), ('tanh',15))
    t = time()-t
    err = elm.loo_press(x,y)    
    
    #yh = elm.predict(x).ravel()
    #from matplotlib import pyplot as plt
    #plt.plot(x,y,'.b')
    #plt.plot(x,yh,'.r')    
    #plt.show()    
    
    print "LOO sine: %.3f at %.3f sec" % (err, t)

def run_iris():
    path = os.path.join(datasets, "classification_iris")
    elm = ELM("classification")
    x = os.path.join(path, "iris_data.txt")
    y = os.path.join(path, "iris_classes.txt")

    t = time()
    elm.train(x,y)
    t = time()-t
    yh = elm.predict(x)
    err = np.sum(np.argmax(yh,1) == np.argmax(np.loadtxt(y),1)) * 1.0 / 150
    print "ACC iris: %.3f at %.3f sec" % (err, t)

def run_mnist():
    path = os.path.join(datasets, "classification_mnist")
    elm = ELM("classification")
    xtr = os.path.join(path, "mnist_Xtr.h5")
    xts = os.path.join(path, "mnist_Xts.h5")
    ytr = os.path.join(path, "mnist_Ytr.h5")
    yts = os.path.join(path, "mnist_Yts.h5")

    t = time()
    elm.train(xtr,ytr)
    t = time()-t

    yh = elm.predict(xts)
    yts = ELM.h5read(yts).ravel()
    err = np.sum(yh == yts) * 1.0 / yts.shape[0]
    print "ACC MNIST: %.3f at %.3f sec" % (err, t)

def run_song():
    path = os.path.join(datasets, "regression_song_year")
    elm = ELM()
    xtr = os.path.join(path, "Xtr.h5")
    xts = os.path.join(path, "Xts.h5")
    ytr = os.path.join(path, "Ytr.h5")
    yts = os.path.join(path, "Yts.h5")

    t = time()
    elm.train(xtr, ytr, [None, 25], ['tanh', 1000])
    t = time()-t

    err = elm.loo_press(xts,yts)
    print "LOO song: %.3f at %.3f sec" % (err, t)


run_sine()
run_iris()
run_mnist()
run_song()