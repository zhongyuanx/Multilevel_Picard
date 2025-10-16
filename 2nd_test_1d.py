#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:58:54 2024

@author: yuanzhong
"""

import numpy as np
import scipy as sp

import rbm

h = 2
beta = 0.1
z = 1
gamma = -1
epsilon = 0.1
ub = -np.log(epsilon*beta)/beta
#ub=1
dim = 1
push_cost = 1
drift_b = 1

def cost(state):
    return h*state

def my_C(beta, lb, ub): 
    return np.sqrt(np.pi/beta)*(1-2*sp.stats.norm.cdf(-np.sqrt(2*beta*(ub-lb))))

def my_g(v): 
    return drift_b*np.minimum(push_cost*np.ones(len(v)) - v, 0) #if v > push_cost else 0

def picard_iter(v1, v2):
    return my_g(v1)-my_g(v2)

alpha = np.sqrt(gamma*gamma+2*beta)+gamma

def DV1(z):
    return h/beta*(1-np.exp(-alpha*z))
V = h*z/beta+h*gamma/beta/beta+h*np.exp(-alpha*z)/beta/alpha

def mlp_v(t, T, z, gamma, level, M):
    if level==0:
        return np.zeros(dim+1)
    
    output = np.zeros(dim+1)
    
    ns = np.power(M, level) 
    S_array = np.random.exponential(scale=1/beta, size=ns)
    tau_array = np.array([rbm.bm_hitting(z, gamma, t) for i in range(ns)])
    temp = np.array([rbm.stopped_rbm(z, gamma, tau_array[i], S_array[i], t) for i in range(ns)])
    output = 1/beta/ns*np.inner(cost(temp[:,1]),np.vstack([np.ones(ns), temp[:,0]/(S_array-t)]))   
    #output = 1/beta/ns*np.inner(cost(temp[:,1])+my_g(DV1(temp[:,1])),np.vstack([np.ones(ns), temp[:,0]/(S_array-t)]))
    #print(S_array, temp, output)
    
    return output

print(mlp_v(0, ub, z, gamma, 1, 100000))
