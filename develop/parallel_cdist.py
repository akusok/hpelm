# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:48:46 2015

@author: akusok
"""

import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
from time import time, sleep


def f(a):
    h, w, ix = a
    return cdist(h, w, "sqeuclidean"), ix

# @profile
def run():
    H = np.random.rand(2,300)
    W = np.random.rand(3000, 300)

    t = time()
    C1 = cdist(H, W, "sqeuclidean")
    print time() - t

    print "done 1"

    t = time()
    k = cpu_count()
    N = H.shape[0]
    idxs = np.array_split(np.arange(N), k*10)
    jobs = [(H[ix], W, ix) for ix in idxs]

    p = Pool(k)
    C1p = np.empty((N, W.shape[0]))
#    for h, w, ix in jobs:
#        C1p[ix] = cdist(h, w, "sqeuclidean")
    t2 = time()
    for h0, ix in p.imap(f, jobs):
        C1p[ix] = h0
    print time() - t2
    p.close()
    print time() - t
    assert np.allclose(C1, C1p)

#    C = cdist(H, W, "cityblock")
#    C = cdist(H, W, "chebyshev")
    


# @profile
def run_all():
    c = run()
    print "Done"


run_all()







