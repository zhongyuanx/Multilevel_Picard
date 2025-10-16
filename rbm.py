#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:13:24 2024

@author: yuanzhong
"""

import numpy as np
import scipy as sp

def my_rtime(beta, lb=0, ub=np.inf): 
    #simulate random sampling times
    temp = lb+np.power(sp.stats.norm.ppf(0.5*(1-np.random.uniform()*(1-2*sp.stats.norm.cdf(-np.sqrt(2*beta*(ub-lb)))))),2)/2/beta
    return temp

def bm_hitting(z, gamma, sigma=1, elapsed=0):
    # simulate tau = hitting time of brownian motion with initial state z, drift gamma and variance sigma^2 to zero starting at elapsed
    # note that the hitting time density is an inverse Gaussian with parameters -z/gamma and z^2/sigma^2
    return elapsed+sp.stats.invgauss.rvs(-sigma**2/z/gamma, scale = z**2/sigma**2)

def stopped_rbm(z, gamma, tau, time, sigma=1, elapsed=0):
    # tau can only be hitting time to zero here; needs to be very specific.
    rs_time = time-elapsed
    rs_tau = tau-elapsed
    
    if rs_time >= rs_tau:
        my_bm_scaled = -(z+gamma*rs_tau)/sigma/np.sqrt(rs_time)
        my_bm_rs_time = np.random.normal(-gamma*(rs_time-rs_tau), np.sqrt(rs_time-rs_tau)*sigma)
        my_rbm = np.sqrt(my_bm_rs_time**2-2*(sigma**2)*(rs_time-rs_tau)*np.log(np.random.uniform()))/2-my_bm_rs_time/2
    
    else:
        std_Gauss = np.random.normal()
        std_Exp = np.random.exponential()
        my_rbm = np.sqrt(2*(sigma**2)*rs_time*(rs_tau-rs_time)/rs_tau*std_Exp+np.power((rs_tau-rs_time)*z/rs_tau+sigma*np.sqrt(rs_time*(rs_tau-rs_time)/rs_tau)*std_Gauss,2))
        my_bm_scaled = ((z**2)*((rs_time**1.5)/(rs_tau**2)-2*np.sqrt(rs_time)/rs_tau)+2*z*sigma*(((rs_tau-rs_time)/rs_tau)**1.5)*std_Gauss+sigma**2*(rs_tau-rs_time)/rs_tau*np.sqrt(rs_time)*(std_Gauss**2+2*std_Exp))/(my_rbm+z)/sigma-gamma*np.sqrt(rs_time)/sigma
        
    return my_bm_scaled, my_rbm

# def rbm0(gamma, time, sigma=1):
#     #inverse transform method to generate instance of rbm0 with drift gamma and variance sigma at given time.
#     my_bm = np.random.normal(-gamma*time, np.sqrt(time)*sigma)
#     return np.sqrt(my_bm**2-2*(sigma**2)*time*np.log(np.random.uniform()))/2-my_bm/2

# def bm_ex(z, tau, time, sigma=1):
#     #brownian excursion constrained to be positive from time 0 to tau
#     #equals z at time 0 and 0 at time tau. simulation for value at given time < tau. 
#     my_mu = (tau-time)*z/tau
#     my_var = (sigma**2)*time*(tau-time)/tau
#     return np.sqrt(2*my_var*np.random.exponential()+np.power(np.random.normal(my_mu, np.sqrt(my_var)),2))

# def stopped_rbm(z, gamma, tau, time, sigma=1, elapsed=0):
#     # tau can only be hitting time to zero here; needs to be very specific.
#     rs_time = time-elapsed
#     rs_tau = tau-elapsed

#     if rs_time >= rs_tau:
#         my_bm = -(z+gamma*rs_tau)/sigma
#         my_rbm = rbm0(gamma, rs_time-rs_tau, sigma)
    
#     else:
#         my_rbm = bm_ex(z, rs_tau, rs_time, sigma)
#         my_bm = (my_rbm-z-gamma*rs_time)/sigma
        
#     return my_bm, my_rbm
    #return np.array([my_bm, my_rbm])



# def normal_density(x):
#     #standard normal density
#     return np.exp(-x*x*0.5)/np.sqrt(2*np.pi)

# def hitting_density(z, gamma, sigma, s):
#     #hitting time density of brownian motion starting at z with drift gamma
#     return z*np.power(s, -1.5)/sigma*normal_density((z+gamma*s)/np.sqrt(s)/sigma)

# def scaled_cond_bm_density(mu, x):
#     #density function formula for conditional density of X(t) given tau=s; general expression
#     return x*(normal_density(x-mu)-normal_density(x+mu))/mu

# def ar_sub(c, density1, density2):
#     #this is subroutine within acceptance rejection
#     my_unif = np.random.uniform()
#     return 1 if my_unif <= density1/c/density2 else 0



# def truncated_normal(mu, sigma):
#     #generate truncated normal>0 with mean mu and standard deviation sigma 
#     y = np.random.normal()*sigma + mu
#     while y<=0:
#         y = np.random.normal()*sigma + mu
    
#     return y

# def bm(z, gamma, tau, time, sigma=1):
#     my_mu = (tau-time)*z/tau
#     my_var = sigma*sigma*time*(tau-time)/tau
#     return np.sqrt(2*my_var*np.random.exponential()+np.power(my_mu+np.sqrt(my_var)*np.random.normal(),2))

# # def bm(z, gamma, tau, time):
# #     #generate instance of X(t) given tau with t < tau, initial state z and drift gamma
# #     my_tmu = np.sqrt((tau-time)/tau/time)*z
# #     my_sigma = np.sqrt((tau-time)*time/tau)
    
# #     #generate a truncated normal with mean my_tmu and standard deviation 
# #     y = truncated_normal(my_tmu, np.sqrt(2))
    
# #     while ar_sub(17, scaled_cond_bm_density(my_tmu, y), normal_density((y-my_tmu)/np.sqrt(2)))==0:
# #         y = truncated_normal(my_tmu, np.sqrt(2))
    
# #     return my_sigma*y


    
# # def bm_hitting(z, gamma, elapsed=0): 
# #     # simulate tau = hitting time of brownian motion with initial state z and drift gamma to zero 
# #     c = np.exp(-z*gamma)
# #     y = np.power(z/np.random.normal(),2)
# #     while ar_sub(c, hitting_density(z, gamma, sigma, y),hitting_density(z, 0, sigma, y))==0:
# #         y = np.power(z/np.random.normal(),2)
    
# #     if y<=1e-8:
# #         print("my hitting time is", y, " and my initial state is ", z, ".\n")
# #     return y+elapsed

# def my_rtime(beta, lb=0, ub=np.inf): 
#     temp = lb+np.power(sp.stats.norm.ppf(0.5*(1-np.random.uniform()*(1-2*sp.stats.norm.cdf(-np.sqrt(2*beta*(ub-lb)))))),2)/2/beta
#     #if temp-lb <= 1e-8:
#         #print("my random time is ", temp-lb, ".\n")
#     return temp
#     # my_S = np.power(sp.stats.norm.ppf(np.random.uniform()/2),2)/2/beta
#     # while my_S > ub-lb:
#     #     #print(my_S)
#     #     my_S = np.power(sp.stats.norm.ppf(np.random.uniform()/2),2)/2/beta
#     # return my_S
    
# def my_rtime2(beta, lb=0, ub=np.inf):
#     return -1/beta*np.log(1-np.random.uniform()*(1-np.exp(-beta*(ub-lb))))+lb



#check for correctness


    