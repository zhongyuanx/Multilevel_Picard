# Multilevel_Picard

This project contains codes for the paper "Multilevel Picard scheme for solving high-dimensional drift control problems with state constraints" by Yuan Zhong, which can be accessed at the link: https://arxiv.org/pdf/2510.21607. 

The main goals of these pieces of codes are to implement an exact simulation version of the multi-level Picard scheme for the test problems in Section 7 of the paper, the least-control policy, and a version of the multi-level Picard scheme using Euler scheme. 

Problem parameters are contained in "PSS_Ex.jl." 

"PSS_Simulations.jl" implements the least control and computes the associated value function estimate. 

"PSS_mlp.jl" implements the multilevel Picard scheme with exact simulation for the test problems described in Section 7 of the paper. It contains a multithread implementation that makes use of all performance cores of the computer. The implementation requires input parameters from "PSS_Ex.jl" and exact simulation of the triple (88) of the paper from "rbm2.jl." 

"eulerdiffusionderivative.jl" implements an Euler scheme for simulating the sample path and the derivative process of the alternative reference process in Section 7 of the paper. Algorithms for the general case can be found in the paper "A Monte Carlo method for estimating sensitivities of reflected diffusions in convex polyhedral domains" by Lipshutz and Ramanan (2019), Stochastic Systems.

"general_mlp_deriv.jl" makes use of "eulerdiffusionderivative.jl" to implement a multilevel Picard scheme for the 2-dimensional example in the paper. 

"pss_output.jl" outputs the data used to plot Figure 2 in the paper. 

"rbm2.jl" implements the exact simulation algorithm found in Appendix D of the paper. 


