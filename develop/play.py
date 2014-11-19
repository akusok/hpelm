# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 23:21:09 2014

@author: Anton
"""

import numpy as np


f = np.sin
f2 = np.vectorize(f)

print type(f)
print type(f2)

f3 = np.vectorize(f2)


print "done"