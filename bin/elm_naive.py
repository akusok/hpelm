#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 16:29:09 2014

@author: akusok
"""

import numpy as np
import os
from hpelm import ELM_Naive

curdir = os.path.dirname(__file__)
pX = os.path.join(curdir, "../datasets/iris/iris_data.npy")
pY = os.path.join(curdir, "../datasets/iris/iris_targets.npy")

X = np.load(pX)
Y = np.load(pY)

Yh = ELM_Naive(X, Y, 20, classification=True)
acc = float(np.sum(Y.argmax(1) == Yh)) / Y.shape[0]
print "%.1f%%" % (acc*100)