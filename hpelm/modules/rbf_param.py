# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:10:02 2015

@author: akusok
"""

from tables import open_file
from six import string_types
from six.moves import xrange
import numpy as np
from scipy.spatial.distance import cdist

# TODO: use these parameters, or remove them

def rbf_param(data, k, kind="sqeuclidean"):
    """Calculates parameters for RBF neurons.

    :param data: - a matrix or an HDF5 file
    """
    if "l1" in kind:
        kind = "cityblock"
    elif "inf" in kind:
        kind = "chebyshev"
    else:
        kind = "sqeuclidean"
    if isinstance(data, string_types):
        h5 = open_file(data, "r")
        X = h5.root.data
    else:
        X = np.array(data)
        assert len(X.shape) == 2, "Data must be a 2-dim matrix"
    N = X.shape[0]
    Nk = min(10*k, N-1)
    dist = np.zeros((Nk,))

    ix = np.random.choice(N, size=Nk)
    for i in xrange(Nk):
        j = ix[i]
        if i == j:
            j += 1
        dist[i] = cdist(X[i][None, :], X[j][None, :], kind)
    m = dist.mean()
    s = dist.std()

    # fill centroids as random points
    ix = np.random.choice(N, size=k)
    W = np.empty((X.shape[1], k))
    for i in range(len(ix)):
        W[:, i] = X[ix[i]]

    # fill bias
    B = np.zeros((k,))
    i = 0
    while True:
        b0 = (np.random.rand()-0.5)*2*s + m
        if b0 > 0:
            B[i] = b0
            i += 1
            if i == k:
                break
    if isinstance(data, string_types):
        h5.close()

    return W, B
