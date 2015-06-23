# -*- coding: utf-8 -*-
"""
Greville and OPIUM method from:
J. Tapson and A. van Schaik, 
"Learning the Pseudoinverse Solution to Network Weights"
Neural Networks

Created on Sun Aug  5 10:05:29 2012

@author: andrevanschaik
"""
from numpy import dot, exp, eye, sqrt

def Greville(x,ee,M,P): #Greville's method for recursive pseudoinverse
    psi = dot(P,x)                      
    nrm1 = 1+dot(x.T,psi)                
    P -= dot(psi,psi.T)/nrm1
    M += dot(ee,psi.T)/nrm1 

def OPIUM(x,ee,M,P,alpha): #OPIUM modification to Greville
    psi = dot(P,x)                      
    nrm1 = 1+dot(x.T,psi)   
    nrm2 = 1+alpha*(1-exp(-sqrt(dot(ee.T,ee))/ee.size))
    P -= dot(psi,psi.T)/nrm1
    P += alpha * eye(P.size**0.5) * (1-exp(-sqrt(dot(ee.T,ee))/ee.size))
    P /= nrm2
    M += dot(ee,psi.T)/nrm1 


