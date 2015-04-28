# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:34:54 2015

@author: akusok
"""


import numpy as np
import os
import hpelm


def classification(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, 'c')
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def regression(folder, nn):
    folder = os.path.join(os.path.dirname(__file__), folder)
    mse = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, "sigm")
        elm.train(Xtr, Ttr)
        Yts = elm.predict(Xts)
        # evaluate classification results
        mse[i - 1] = np.mean((Tts - Yts) ** 2)
    return mse


def classification_v(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # create validation set
        N = Xtr.shape[0]
        ix = np.arange(N)
        np.random.shuffle(ix)
        Xvl = Xtr[ix[:N/5]]
        Tvl = Ttr[ix[:N/5]]
        Xtr = Xtr[ix[N/5:]]
        Ttr = Ttr[ix[N/5:]]
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "c", "V", Xv=Xvl, Tv=Tvl)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def regression_v(folder, nn):
    folder = os.path.join(os.path.dirname(__file__), folder)
    mse = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # create validation set
        N = Xtr.shape[0]
        ix = np.arange(N)
        np.random.shuffle(ix)
        Xvl = Xtr[ix[:N/5]]
        Tvl = Ttr[ix[:N/5]]
        Xtr = Xtr[ix[N/5:]]
        Ttr = Ttr[ix[N/5:]]
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, "sigm")
        elm.train(Xtr, Ttr, "V", Xv=Xvl, Tv=Tvl)
        Yts = elm.predict(Xts)
        # evaluate classification results
        mse[i - 1] = np.mean((Tts - Yts) ** 2)
    return mse


def classification_cv(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "c", "CV", k=5)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def regression_cv(folder, nn):
    folder = os.path.join(os.path.dirname(__file__), folder)
    mse = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, "sigm")
        elm.train(Xtr, Ttr, "CV", k=5)
        Yts = elm.predict(Xts)
        # evaluate classification results
        mse[i - 1] = np.mean((Tts - Yts) ** 2)
    return mse


def classification_loo(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "c", "LOO")
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def regression_loo(folder, nn):
    folder = os.path.join(os.path.dirname(__file__), folder)
    mse = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, "sigm")
        elm.train(Xtr, Ttr, "LOO")
        Yts = elm.predict(Xts)
        # evaluate classification results
        mse[i - 1] = np.mean((Tts - Yts) ** 2)
    return mse


def classification_mc_loo(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "mc", "LOO")
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def classification_mc_cv(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "mc", "CV", k=5)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def classification_wc_loo(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        w = np.ones((Ttr.shape[1],))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "wc", "LOO", w=w)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc


def classification_wc_cv(folder, nn, ntype="sigm"):
    folder = os.path.join(os.path.dirname(__file__), folder)
    acc = np.empty((10,))
    for i in range(10):  # 10 random initializations
        # get file names
        Xtr = np.load(os.path.join(folder, "xtrain_%d.npy" % (i + 1)))
        Xts = np.load(os.path.join(folder, "xtest_%d.npy" % (i + 1)))
        Ttr = np.load(os.path.join(folder, "ytrain_%d.npy" % (i + 1)))
        Tts = np.load(os.path.join(folder, "ytest_%d.npy" % (i + 1)))
        w = np.ones((Ttr.shape[1],))
        # train ELM
        elm = hpelm.ELM(Xtr.shape[1], Ttr.shape[1])
        elm.add_neurons(nn, ntype)
        elm.train(Xtr, Ttr, "wc", "CV", k=5, w=w)
        Yts = elm.predict(Xts)
        # evaluate classification results
        Tts = np.argmax(Tts, 1)
        Yts = np.argmax(Yts, 1)
        acc[i - 1] = float(np.sum(Yts == Tts)) / Tts.shape[0]
    return acc
