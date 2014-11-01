# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:50:54 2014

@author: akusok
"""

import numpy as np
import time


@profile
def line_count(filename):
    start_time = time.time()

    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    print lines, time.time() - start_time


@profile
def run():
    
    #x =np.random.randn(100000000,5)
    x = np.random.randn(10000000,5)

    b = 3952
    n = x.shape[0]
    
    E_x = 0
    E_x2 = 0
    
    for i in range(n/b + 1):
        k = min(b, n-i*b)
        xb = x[i*b:i*b+k]
        E_x += np.mean(xb,0) * (1.0*k/n)    
        E_x2 += np.mean(xb**2,0) * (1.0*k/n)    
            
    E2_x = E_x**2
    sh = (E_x2 - E2_x)**0.5

    s = np.std(x,0)
    
    print sh
    print s


#run()
bufcount("/home/akusok/Documents/X-ELM/hpelm/datasets/regression_song_year/Xtr.txt")








