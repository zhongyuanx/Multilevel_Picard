#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:47:15 2024

@author: yuanzhong
"""

import numpy as np

beta = 0.1 #discount factor
holding_cost = 2

T=1
v=0
K = 1



def my_g(K, v):
    return (1-v)*K if v>1 else 0

def picard_iter(K, v1, v2):
    return my_g(k,v1)-my_g(k,v2)

def cost(holding_cost, z):
    return holding_cost*z

def mlp_v(z, t, T, level, M, mesh):
        