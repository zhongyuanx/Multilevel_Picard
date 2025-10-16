#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:06:01 2023

@author: yuanzhong
"""

nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]
k = 3

def longestOnes(nums, k):
    length = 0 # current max length
    current = 0 # length of longest ending at current position
    count = 0 # number of zeros in longest ending at current position
    i=0
    for j in range(len(nums)):
        if nums[j]==1:
            current+=1
            length = max(length, current)
            
        elif count+1<=k: #number of zeros <= k
            count+=1
            current+=1
            length = max(length, current)

        else: # number of zeros > k
            if nums[i]==0:
                i+=1
                current-=1
            else:
                while nums[i]==1:
                    i+=1
                current = j-i
        
        print(length, current, i, j)

    return length

print(longestOnes(nums, k))