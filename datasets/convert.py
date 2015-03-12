# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:01:29 2015

@author: akusok
"""

import numpy as np
from tables import open_file
import os


for root, dirs, _ in os.walk("."):
    break
mydirs = [d for d in dirs if "Unittest" not in d]

for d in mydirs:
    for root, _, files in os.walk("./" + d):
        for f in files:
            if f[-2:] != "h5":
                continue
            h5f = os.path.join(root, f)
            npf = os.path.join(root, f[:-2] + "npy")
            h5 = open_file(h5f)
            for node in h5.walk_nodes():
                pass
            data = node[:].T
            np.save(npf, data)
            h5.close()
            print h5f

mydirs = [d for d in dirs if "Classification" in d]
for d in mydirs:
    for root, _, files in os.walk("./" + d):
        for f in files:
            if f[0] == "y" and f[-1] == "5":
                h5f = os.path.join(root, f)
                npf = os.path.join(root, f[:-2] + "npy")
                h5 = open_file(h5f)
                for node in h5.walk_nodes():
                    pass
                data = node[:].astype(np.int).T
                c = data.max()
                if c > 1:  # convert multiclass cases
                    print h5f
                    if data.min() > 0:
                        data = data - data.min()
                    n = data.shape[0]
                    data2 = np.zeros((n, c), np.int)
                    for i in xrange(n):
                        data2[i, data[i]] = 1
                    np.save(npf, data2)
                h5.close()
