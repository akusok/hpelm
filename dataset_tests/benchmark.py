# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:15:12 2015

@author: akusok
"""

import numpy as np
import os
import hpelm
import cPickle
import sys
from time import time


def elm(folder, i, nn, param):
#    folder = os.path.join(os.path.dirname(__file__), folder)
#    acc = np.empty((10, 3))

    # get file names
    Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
    Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
    Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
    Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))

    # create validation set
#    N = Xtr.shape[0]
#    ix = np.arange(N)
#    np.random.shuffle(ix)
#    Xvl = Xtr[ix[:N/5]]
#    Tvl = Ttr[ix[:N/5]]
#    Xtr = Xtr[ix[N/5:]]
#    Ttr = Ttr[ix[N/5:]]

#    elm.add_neurons(Xtr.shape[1], "lin")
#    W, B = hpelm.modules.rbf_param(Xtr, nn, "l2")
#    elm.add_neurons(nn, "rbf_l2", W, B)

    nn = min(nn, Xtr.shape[0]/2)

    t = time()
    # build ELM
    elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
    elm.add_neurons(nn, "sigm")
    # train ELM
    elm.train(Xtr, Ttr, *param)
    Yts = elm.predict(Xts)
    err = elm.error(Yts, Tts)
    t = time() - t

    nns = [l[1] for l in elm.neurons]
    return err, nns, t


def trainer(folder, cls=False):
    nn = 100
    errs = np.zeros((10, 3))
    neurs = np.zeros((10, 3), dtype=np.int)
    times = np.zeros((10, 3))

    if cls:
        param = ['c']
    else:
        param = []

    for i in xrange(10):
        print i
        e, l, t = elm(folder, i, nn, param)
        errs[i, 0] = e
        neurs[i, 0] = l[0]
        times[i, 0] = t
        e, l, t = elm(folder, i, nn, param+['loo'])
        errs[i, 1] = e
        neurs[i, 1] = l[0]
        times[i, 1] = t
        e, l, t = elm(folder, i, nn, param+['OP', 'loo'])
        errs[i, 2] = e
        neurs[i, 2] = l[0]
        times[i, 2] = t

    stds = errs.std(0)
    errs = errs.mean(0)
    neurs = neurs.mean(0)
    times = times.mean(0)
    fname = folder+".pkl"
    cPickle.dump((errs, stds, neurs, times), open(fname, "wb"), -1)


if __name__ == "__main__":
    datas = (("Classification-Iris", True),
             ("Classification-Pima_Indians_Diabetes", True),
             ("Classification-Wine", True),
             ("Classification-Wisconsin_Breast_Cancer", True),
             ("Regression-Abalone", False),
             ("Regression-Ailerons", False),
             ("Regression-Auto_price", False),
             ("Regression-Bank", False),
             ("Regression-Boston", False),
             ("Regression-Breast_cancer", False),
             ("Regression-Computer", False),
             ("Regression-CPU", False),
             ("Regression-Elevators", False),
             ("Regression-Servo", False),
             ("Regression-Stocks", False))

    j = int(sys.argv[1])
    f1, c1 = datas[j]
    print f1
    f1 = "/home/akusok/Dropbox/Documents/X-ELM/hpelm/dataset_tests/" + f1
    trainer(f1, c1)



