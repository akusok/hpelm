# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:05:03 2014

@author: akusoka1
"""

import numpy as np
from f_apply import f_apply


n = 100000
k = 2000
H = np.ones((n,k))
f = np.random.randint(0,2,size=(k,)).astype(np.int32)



f_apply(H,f)
print f
print H[0]