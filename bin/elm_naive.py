#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 16:29:09 2014

@author: akusok
"""

import numpy as np
import os
from hpelm import ELM

curdir = os.path.dirname(__file__)
pX = os.path.join(curdir, "../datasets/Unittest-Iris/iris_data.txt")
pY = os.path.join(curdir, "../datasets/Unittest-Iris/iris_classes.txt")

X = np.loadtxt(pX)
Y = np.loadtxt(pY)

elm = ELM(4,3)
elm.add_neurons(15, "sigm")
elm.train(X, Y, "c")
Yh = elm.predict(X)
acc = float(np.sum(Y.argmax(1) == Yh.argmax(1))) / Y.shape[0]
print("Iris dataset training error: %.1f%%" % (100-acc*100))