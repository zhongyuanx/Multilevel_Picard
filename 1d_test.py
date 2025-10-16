#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:18:51 2024

@author: yuanzhong
"""

import numpy as np
import scipy as sp

import rbm

offset = 2.5
h = 2
beta = 1
z = 1
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

def my_C2(beta, lb, ub):
    return 1/beta*(1-np.exp(-beta*(ub-lb)))

def my_g(v): 
    return offset*v+drift_b*np.minimum(push_cost*np.ones(len(v)) - v, 0) #if v > push_cost else 0
    #return drift_b*np.maximum(np.minimum(push_cost*np.ones(len(v)) - v, 0),-1) #if v > push_cost else 0

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
    
    # random times
    S_array = np.array([rbm.my_rtime(beta, t, T) for i in range(ns)])
    #S_array = np.array([rbm.my_rtime2(beta, t, T) for i in range(ns)])
    
    # hitting times
    tau_array = np.array([rbm.bm_hitting(z, gamma, sigma, t) for i in range(ns)])
    
    # R(S) and B(S\wedge \tau)
    temp = np.array([rbm.stopped_rbm(z, gamma, tau_array[i], S_array[i], sigma, t) for i in range(ns)])
    
    # the result
    incr = my_C(beta, t, T)/ns*np.inner(cost(temp[:,1]),np.vstack([np.ones(ns)*np.sqrt(S_array-t), temp[:,0]/np.sqrt(S_array-t)/sigma]))
    #incr = my_C2(beta, t, T)/ns*np.inner(cost(temp[:,1]),np.vstack([np.ones(ns), temp[:,0]/(S_array-t)]))
    
    output+=incr
    #if (t==0 and z==1):
    #    print(output, "0")
    # for i in range(ns):
    #     if tau_array[i]<=1e-8 or S_array[i]<=1e-8 or temp[i][1]<=1e-6:
    #         print(temp[i][1], tau_array[i], S_array[i], temp[i][0])
    #print(S_array, temp, output)

    # for i in range(ns):
    #     my_S = rbm.my_rtime(beta, t, T)
    #     my_tau = rbm.bm_hitting(z, gamma, t)
    #     my_B, my_R = rbm.stopped_rbm(z, gamma, my_tau, my_S, t)
    #     output += cost(my_R)*np.array([np.sqrt(my_S-t), my_B/np.sqrt(my_S-t)])
    # output = output/ns*my_C(beta, t, T)
    #print(output)
    #print(output[0], level)
    

    #for l in range(level-1,0,-1):
    for l in range(1,level):
        # number of simulated instances
        ns = np.power(M, level-l)
        
        # random times
        S_array = np.array([rbm.my_rtime(beta, t, T) for i in range(ns)])
        #S_array = np.array([rbm.my_rtime2(beta, t, T) for i in range(ns)])
        
        # hitting times
        tau_array = np.array([rbm.bm_hitting(z, gamma, sigma, t) for i in range(ns)])
        
        # R(S) and B(S\wedge \tau)
        temp = np.array([rbm.stopped_rbm(z, gamma, tau_array[i], S_array[i], sigma, t) for i in range(ns)])
        
        #v at level l and v at level l-1
        #declare temp[i][1] to have hit zero if it is small enough. 
        v1 = np.array([mlp_v(S_array[i], T, temp[i][1], gamma, sigma, l, M)[1] if temp[i][1]>1e-6 else 0 for i in range(ns)])
        v2 = np.array([mlp_v(S_array[i], T, temp[i][1], gamma, sigma, l-1, M)[1] if temp[i][1]>1e-6 else 0 for i in range(ns)])
        
        # increment to result
        incr = my_C(beta, t, T)/ns*np.inner(picard_iter(v1, v2), np.vstack([np.ones(ns)*np.sqrt(S_array-t), temp[:,0]/np.sqrt(S_array-t)/sigma]))
        #incr = my_C2(beta, t, T)/ns*np.inner(picard_iter(v1, v2), np.vstack([np.ones(ns), temp[:,0]/(S_array-t)]))
        
        # if l==4:
        #     print("incr at level ", l, "=", incr)
        output+=incr
        #if t==0 and z==1:
        #    print(output, l)
        
        # temp = np.zeros(dim+1)
        # for i in range(ns):
        #     my_S = rbm.my_rtime(beta, t, T)
        #     my_tau = rbm.bm_hitting(z, gamma, t)
        #     print(my_S, my_tau, t)
        #     my_B, my_R = rbm.stopped_rbm(z, gamma, my_tau, my_S, t)
        #     #print(len(bm), np.average(bm))
        #     v1 = mlp_v(my_S, T, my_R, gamma, l, M)
        #     v2 = mlp_v(my_S, T, my_R, gamma, l-1, M)
        #     temp+=picard_iter(v1[1], v2[1])*np.array([np.sqrt(my_S-t), my_B/np.sqrt(my_S-t)])
        #     #print("test", rtime, v1[0], v2[0], picard_iter(v1,v2))
        # #print(ns)
        # temp = temp/ns*my_C(beta, t, T)
        # output+=temp
        
    #print(output, my_f(output), level)
    return output

alpha = (np.sqrt(gamma**2+2*beta*(sigma**2))+gamma)/(sigma**2)

def DV1(z):
    return h/beta*(1-np.exp(-alpha*z))
V = h*z/beta+h*gamma/beta/beta+h*np.exp(-alpha*z)/beta/alpha

print("T=", ub, "; sigma=", sigma)
print(V, DV1(z))

#print(mlp_v(0, ub, z, gamma, sigma, 1, 100000))
#print(mlp_v(0, ub, z, gamma, sigma, 2, 300))
#print(mlp_v(0, ub, z, gamma, sigma, 3, 30))
# print(mlp_v(0, ub, z, gamma, 4, 10))
for i in range(5):
    print(mlp_v(0, ub, z, gamma, sigma, 5, 10))
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
