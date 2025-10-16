#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:29:04 2022

@author: yuanzhong
"""

"""
This is the code for the benchmark where 
the nonlinear term is simply -u. test1.py gives the solution for it.
"""

import numpy as np

T = 1
dim = 100
x = np.zeros(dim)
#x = np.ones(dim)

def my_g(y):
    return np.log(1+sum(y[:]*y[:]))

def mlp_v(t, T, x, level, M):
    if level==0 or level==-1:
        return 0
    
    output = 0
    temp = 0
    ns = np.power(M, level)
    z = np.random.normal(size=(ns,len(x)))
    for i in range(ns):
        temp+=my_g(x+np.sqrt(T-t)*z[i,:])
    temp = temp/ns
    output+=temp
    
    for l in range(level):
        ns = np.power(M, level-l)
        #generate random times
        rtime = np.random.uniform(t, T, size=ns)
        #generate normals
        bm = np.random.normal(size=(ns,len(x)))
        temp = 0
        for j in range(ns):
            temp-=mlp_v(rtime[j],T,x+bm[j,:]*np.sqrt(rtime[j]-t),l,M)
            #print(mlp_v(rtime[j],T,x+bm[j]*np.sqrt(rtime[j]-t),l,M), l, M)
            if l>0:
                temp+=mlp_v(rtime[j],T,x+bm[j,:]*np.sqrt(rtime[j]-t),l-1,M)
        temp = temp/ns*(T-t)
        output+=temp
    
    return output

for i in range(10):
    print(mlp_v(0, 1, x, 5, 5))