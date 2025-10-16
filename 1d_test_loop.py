#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:18:51 2024

@author: yuanzhong
"""

import numpy as np
import scipy as sp

import rbm

offset = 0
h = 2
beta = 1
z = 0.5
gamma = -1-offset
epsilon = 0.01
ub = -np.log(epsilon*beta)/beta
#ub=1
dim = 1
push_cost = 1
drift_b = 5
sigma = 1

def cost(state):
    return h*state

def my_C(beta, lb, ub): 
    return np.sqrt(np.pi/beta)*(1-2*sp.stats.norm.cdf(-np.sqrt(2*beta*(ub-lb))))

def my_g(v): 
    return offset*v+drift_b*np.minimum(push_cost - v, 0) #if v > push_cost else 0

def picard_iter(v1, v2):
    return my_g(v1)-my_g(v2)

def mlp_v(t, T, z, gamma, sigma, level, M):
    output = np.zeros(dim+1)
    #print(t, x, level)
    if level==0:
        return output
        #return np.zeros(dim+1)
        
    # number of simulated instances
    ns = np.power(M, level) 
    
    for i in range(ns):
        my_S = rbm.my_rtime(beta, t, T)
        my_tau = rbm.bm_hitting(z, gamma, sigma, t)
        my_B, my_R = rbm.stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
        output += cost(my_R)*np.array([np.sqrt(my_S-t), my_B/sigma])
    output = output/ns*my_C(beta, t, T)    

    #for l in range(level-1,0,-1):
    for l in range(1,level):
        # number of simulated instances
        ns = np.power(M, level-l)
                
        temp = np.zeros(dim+1)
        for i in range(ns):
            my_S = rbm.my_rtime(beta, t, T)
            my_tau = rbm.bm_hitting(z, gamma, sigma, t)
            my_B, my_R = rbm.stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
            v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
            v2 = mlp_v(my_S, T, my_R, gamma, sigma, l-1, M)
            temp+=picard_iter(v1[1], v2[1])*np.array([np.sqrt(my_S-t), my_B/sigma])
        temp = temp/ns*my_C(beta, t, T)
        output+=temp

    return output

alpha = (np.sqrt(gamma**2+2*beta*(sigma**2))+gamma)/(sigma**2)

def DV1(z):
    return h/beta*(1-np.exp(-alpha*z))
V = h*z/beta+h*gamma/beta/beta+h*np.exp(-alpha*z)/beta/alpha

print("T =", ub, "; sigma =", sigma)
print(V, DV1(z))

#for i in range(3):
    #print(mlp_v(0, ub, z, gamma, sigma, 1, 100000))
    #print(mlp_v(0, ub, z, gamma, sigma, 2, 300))
    #print(mlp_v(0, ub, z, gamma, sigma, 3, 30))
print(mlp_v(0, ub, z, gamma, sigma, 1, 10000))
# print(mlp_v(0, ub, z, gamma, 4, 10))

#print(mlp_v(0, ub, z, gamma, 6, 10))
#print(mlp_v(0, ub, z, gamma, 2, 1000))




# print(mlp_v(0, ub, z, gamma, 1, 1000000))
# print(mlp_v(0, ub, z, gamma, 2, 1000))
# print(mlp_v(0, ub, z, gamma, 3, 100))
# print(mlp_v(0, ub, z, gamma, 4, 32))
# print(mlp_v(0, ub, z, gamma, 5, 16))
# print(mlp_v(0, ub, z, gamma, 6, 10))
# print(mlp_v(0, ub, z, gamma, 7, 7))


# for i in range(1):
#     print(mlp_v(0, ub, z, gamma, 5, 6))

        


#for i in range(1):
#    print(mlp_v(0, ub, z, gamma, 1, 100), DV1) #, V, DV1)



#V0 = h/np.sqrt(2)/np.power(beta, 1.5)

# my_sim = np.zeros(100000)

# for i in range(100000):
#     my_S = rbm.my_rtime(beta, ub=ub)
#     #my_S = rbm.my_rtime(beta)
#     my_tau = rbm.bm_hitting(z, gamma)
#     SB, SR = rbm.stopped_rbm(z, gamma, my_tau, my_S)
#     my_sim[i] = SR*SB/np.sqrt(my_S)

#print(h*np.sqrt(np.pi/beta)*np.average(my_sim), DV1)
#print(h*np.sqrt(np.pi/beta)*(1-2*sp.stats.norm.cdf(-np.sqrt(2*beta*(ub))))*np.average(my_sim), DV1, ub)

# for i in range(100000):
#     my_S = np.random.exponential(1/beta)
#     while my_S > ub:
#         my_S = np.random.exponential(1/beta)
#     my_sim[i] = rbm.rbm0(0, my_S)
    
# print(h*np.average(my_sim)/(1-np.exp(-beta*T)))
