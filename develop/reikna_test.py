# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 08:47:44 2015

@author: akusok
"""

import numpy
from numpy.linalg import norm
import reikna.cluda as cluda
from reikna.linalg import MatrixMul
import time


#@profile
def run():

    api = cluda.ocl_api()
    thr = api.Thread.create()
    
    n = 3000
    shape1 = (n, n)
    shape2 = (n, n)
    
    a = numpy.random.randn(*shape1).astype(numpy.float32)
    b = numpy.random.randn(*shape2).astype(numpy.float32)
    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    res_dev = thr.array((shape1[0], shape2[1]), dtype=numpy.float32)
    
    dot = MatrixMul(a_dev, b_dev, out_arr=res_dev)
    dotc = dot.compile(thr)
    dotc(res_dev, a_dev, b_dev)
    
    res_reference = numpy.dot(a, b)
    
    print(norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6)
    
run()