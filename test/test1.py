#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:29:10 2022

@author: yuanzhong
"""

import numpy as np

T = 1
ns = 100000
dim = 100
x = np.zeros(dim)
#x = np.ones(dim)

def my_g(y):
    return np.log(1+sum(y[:]*y[:]))

def my_v(t, T, x, num_sample):
    z = np.random.normal(size=(num_sample,len(x)))
    total = 0
    s = np.sqrt(T-t)
    for i in range(num_sample):
        total += my_g(x+s*z[i,:])
    return total/num_sample*np.exp(t-T)

print(my_v(0, 1, x, ns))
