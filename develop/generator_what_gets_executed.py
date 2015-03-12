# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:15:27 2014

@author: akusok
"""


def gen(k=None):
    print "starting"
    for i in range(3):
        print "generating i: ", i
        yield i
    print "finalizing", k
    
def gen2():
    d = 1
    return d, gen()    
    

###############################
g = gen()  
print list(g)  # finalizes!

###############################
g = gen()  
for g1 in g:
    print g1  # finalizes!

###############################
d,g = gen2()
print "d = ", d
for g1 in g:
    print g1

###############################
f = gen('f')
g = gen('g')
for a,b in zip(f,g):
    print a,b  # only F finalizes!
     
###############################
print "+++++++++++++++++"
f = gen('f')
g = gen('g')
for a in f:
    b = g.next()
    print a,b  # only F finalizes!





    