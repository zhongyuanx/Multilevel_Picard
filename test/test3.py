#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:36:00 2023

@author: yuanzhong
"""

"""
This is the code for benchmark with explicit solution. 
The nonlinear term involves both the function and its gradient, 
and is not lipschitz.
"""

import numpy as np

T = 10
dim = 10
x = np.zeros(dim)

def my_g(t, y):
    return np.exp(t+np.sum(y))/(1+np.exp(t+np.sum(y)))

def my_f(v):
    return (v[0]-1/len(v[1:])-0.5)*np.sum(v[1:])

def picard_iter(v1, v2):
    #coeff = my_g(t, x)
    #return my_f(coeff, v1[1:])-my_f(coeff, v2[1:])
    return my_f(v1) - my_f(v2) #if l > 0 else 0)
    

def mlp_v(t, T, x, level, M):
    #print(t, x, level)
    if level==0:
        return np.zeros(len(x)+1)
    
    output = np.append([my_g(T, x)], np.zeros(dim))
    temp = np.zeros(len(x)+1)
    ns = np.power(M, level)
    z = np.random.normal(0, np.sqrt(T-t), size=(ns,len(x)))
    for i in range(ns):
        temp+=(my_g(T, x+z[i,:])-my_g(T, x))*np.append([1],z[i,:]/(T-t))
    temp = temp/ns
    output+=temp
    #print(output[0], level)
    

    for l in range(level-1,0,-1):
        ns = np.power(M, level-l)
        #print(l, level, ns)
        #generate random times
        #rtau = np.random.uniform(size=ns)
        #rtime = t+(T-t)*rtau*rtau
        #generate normals
        #bm = np.random.normal(size=(ns,len(x)))
        temp = np.zeros(len(x)+1)
        for i in range(ns):
            rtau = np.random.uniform()
            rtime = t+(T-t)*rtau*rtau
            bm = np.random.normal(size=len(x))
            #print(len(bm), np.average(bm))
            y = x+np.sqrt(rtime-t)*bm
            v1 = mlp_v(rtime, T, y, l, M)
            v2 = mlp_v(rtime, T, y, l-1, M)
            temp+=picard_iter(v1, v2)*np.append([np.sqrt(rtime-t)],bm)
            #print("test", rtime, v1[0], v2[0], picard_iter(v1,v2))
        #print(ns)
        temp = temp*(2*np.sqrt(T-t)/ns)
        output+=temp
        
    #print(output, my_f(output), level)
    return output

print(my_g(0, x)) 

for i in range(10):
    print((mlp_v(0, T, x, 6, 6))[0])

"""
l = []
for i in range(10):
    s = mlp_v(0, T, x, 3, 6)[0]
    l.append(s)
    print(s)

print(sum(l)/len(l))
"""
    
