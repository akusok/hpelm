# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 18:52:32 2014

@author: akusok
"""
import numpy as np


def semi_Tikhonov(H, T, Tmean):
    # add semi-Tikhonov regularization: small random noise projected to "zero"
    # "zero" = zero + E[T], otherwise we introduce a bias
    nT = H.shape[0]/10 + 1
    tkH = np.random.rand(nT, H.shape[1]) * 10E-6
    tkT = np.tile(Tmean, (nT,1))
    H = np.vstack((H, tkH))
    T = np.vstack((T, tkT))
    return H, T






















