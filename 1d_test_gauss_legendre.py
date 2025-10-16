#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:44:54 2024

@author: yuanzhong
"""

import numpy as np
import scipy as sp

import rbm

deg = 10 #degree of Gauss-Legendre quadrature

offset = 1
h = 2
beta = 0.1
z = 1
gamma = -1-offset
epsilon = 0.1
ub = -np.log(epsilon*beta)/beta
#ub=1
dim = 1
push_cost = 1
drift_b = 2

def cost(state):
    return h*state

def my_C(beta, lb, ub): 
    return np.sqrt(np.pi/beta)*(1-2*sp.stats.norm.cdf(-np.sqrt(2*beta*(ub-lb))))

def my_C2(beta, lb, ub):
    return 1/beta*(1-np.exp(-beta*(ub-lb)))

def my_g(v): 
    return offset*v+drift_b*np.minimum(push_cost*np.ones(len(v)) - v, 0) #if v > push_cost else 0
    #return drift_b*np.maximum(np.minimum(push_cost*np.ones(len(v)) - v, 0),-1) #if v > push_cost else 0

def picard_iter(v1, v2):
    return my_g(v1)-my_g(v2)

def gauss_legendre(t, T, deg):
    unscaled_times, unscaled_weights = np.polynomial.legendre.leggauss(deg)
    times = 0.5*(unscaled_times+1)*(T-t)+t
    weights = 0.5*(T-t)*unscaled_weights
    return times, weights

def budgets(times, weights, ns, beta):
    temp = weights*weights*np.exp(-2*beta*times)
    return np.ceil(ns*temp/np.sum(temp)).astype('int')
    

def mlp_v(t, T, z, gamma, level, M, deg):
    output = np.zeros(dim+1)
    #print(t, x, level)
    if level==0:
        return output
        #return np.zeros(dim+1)
    

    
    # number of simulated instances
    ns = np.power(M, level) 

    #time points to estimate and weights
    times, weights = gauss_legendre(t, T, deg)
    my_budget = budgets(times, weights, ns, beta)
    
    for j in range(deg):
        tau_array = np.array([rbm.bm_hitting(z, gamma, t) for i in range(my_budget[j])])
        temp = np.array([rbm.stopped_rbm(z, gamma, tau_array[i], times[j], t) for i in range(my_budget[j])])
        incr = weights[j]*np.exp(-beta*times[j])*np.inner(cost(temp[:,1]),np.vstack([np.ones(my_budget[j]), temp[:,0]/(times[j]-t)]))/my_budget[j]
        output+=incr
    
    if t==0 and z==1:
        print(output, 0)

    for l in range(1,level):
        # number of simulated instances
        ns = np.power(M, level-l)
        
        for j in range(deg):
            tau_array = np.array([rbm.bm_hitting(z, gamma, t) for i in range(ns)])
            temp = np.array([rbm.stopped_rbm(z, gamma, tau_array[i], times[j], t) for i in range(ns)])
            v1 = np.array([mlp_v(times[j], T, temp[i][1], gamma, l, M, deg)[1] if temp[i][1]>1e-6 else 0 for i in range(ns)])
            v2 = np.array([mlp_v(times[j], T, temp[i][1], gamma, l-1, M, deg)[1] if temp[i][1]>1e-6 else 0 for i in range(ns)])
            incr = weights[j]*np.exp(-beta*times[j])*np.inner(picard_iter(v1, v2), np.vstack([np.ones(ns), temp[:,0]/(times[j]-t)]))/ns    
            output+=incr
    
        if t==0 and z==1:
            print(output, l)

    return output

alpha = np.sqrt(gamma*gamma+2*beta)+gamma

def DV1(z):
    return h/beta*(1-np.exp(-alpha*z))
V = h*z/beta+h*gamma/beta/beta+h*np.exp(-alpha*z)/beta/alpha

print(V, DV1(z))

# print(mlp_v(0, ub, z, gamma, 1, 10000))
# print(mlp_v(0, ub, z, gamma, 2, 100))
# print(mlp_v(0, ub, z, gamma, 3, 30))
# print(mlp_v(0, ub, z, gamma, 4, 10))
# print(mlp_v(0, ub, z, gamma, 5, 6))
#print(mlp_v(0, ub, z, gamma, 6, 10))
print(mlp_v(0, ub, z, gamma, 5, 16, deg))




# print(mlp_v(0, ub, z, gamma, 1, 1000000))
# print(mlp_v(0, ub, z, gamma, 2, 1000))
# print(mlp_v(0, ub, z, gamma, 3, 100))
# print(mlp_v(0, ub, z, gamma, 4, 32))
# print(mlp_v(0, ub, z, gamma, 5, 16))
# print(mlp_v(0, ub, z, gamma, 6, 10))
# print(mlp_v(0, ub, z, gamma, 7, 7))


