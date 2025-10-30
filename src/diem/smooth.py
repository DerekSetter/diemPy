import numba
import numpy as np
#import timeit
#import time
import math
from matplotlib import pyplot as plt
#import pandas as pd
from numba import prange
import multiprocessing as mp



# this function returns the reach, i.e. distance away that we wil look at given the scale
@numba.njit
def TruncatedLaplaceReach(scale): 
    return scale * 2.995732273553991

# this function returns the weight using the truncated laplace (unnormalized, as in stuart's notebook)
@numba.njit
def TruncatedLaplace(x,scale):
    
    if (0<= x) and ( x < scale*2.995732273553991):
        res = 0.5263157894736842 * math.exp( - x / scale)
    elif (x<0) and (-scale * 2.995732273553991 < x):
         res = 0.5263157894736842 * math.exp( x / scale)
    else: res = 0

    return res
    
# this function returns the weights of each site using TruncatedLaplace defined above
@numba.njit
def get_laplace_weights(posArr,focus,scale):
    if len(posArr) < 1: raise Exception("the number of positions must be >=1")
    posArr = posArr - posArr[focus]
    
    #get the weights of each site
    weights = np.zeros(len(posArr))-1.0
    for idx in range(len(weights)):
        weights[idx] = TruncatedLaplace(posArr[idx],scale)

    return weights




# this function gets the weights of all the markers within the window
# it returns the new state after smothing, determined by the max weight or the nearest marker with the max weight (including itself)
# *** this version WILL convert a 'called' marker to an 'uncalled' marker if the surrounding markers are 'uncalled'
@numba.njit
def get_laplace_smoothed_value(posArr,valArr,focus,weights): #this version will convert a called marker to an uncalled marker
    #print('val arr = ', valArr)                         
    res = -1
    if len(valArr)<1: raise Exception(f"value array must be non-empty", f"focus = {focus}", "pos array = ", posArr , "val array = ", valArr, " weights = ", weights) 
    if len(valArr) == 1: #if there are no other informative markers nearby, return the current value
        return valArr[0]
        
    posArr = posArr - posArr[focus] #change to distances relative to focal site

    # create an array with the weights of the markers [0, 1, 2]
    zeroWeight = 0+sum(weights[valArr == 0])
    oneWeight = 0+sum(weights[valArr==1])
    twoWeight = 0+sum(weights[valArr == 2])
    threeWeight = 0+sum(weights[valArr == 3])
    weightArr = np.array([zeroWeight,oneWeight,twoWeight,threeWeight]) 

    # get the states with the higest weight
    mVal = np.max(weightArr)
    mIndex = np.array([0,1,2,3])[weightArr == mVal]

    if len(mIndex == 1): #there is an obvious max
        res = mIndex[0]
    else: # it could go more than one way
        if np.any(mIndex == valArr[focus]): #if the current value could be the max, do not reassign
            res = valArr[focus]
        else: #assign the state of the closest marker with the maximum weight, and if that fails (it shouldn't), do not reassign
            xv = valArr[focus]
            xd = max(posArr)+1
            for m in mIndex:
                minDist = min(posArr[valArr == m])
                if minDist < xd:
                    xd = minDist
                    xv = m
            res = xv
    
    if res == -1: raise Exception("something went wrong with get_laplace_smoothed_value")
        
    return res


@numba.njit
def get_interval_indices(posArr,scale):
    maxDist = TruncatedLaplaceReach(scale)
    #print("scale is ",scale," and max distance is ",maxDist) 
    
    left = 0
    focus = 0
    right = 0
    
    nSites = len(posArr)
    res = np.zeros((nSites,2),dtype=np.int64)
    

    for focusIdx,focusPos in enumerate(posArr):

        while left<=focusIdx and (focusPos - posArr[left])>maxDist: left +=1
        while right< len(posArr)-1 and (posArr[right+1] - focusPos)<maxDist: right += 1

        res[focusIdx] = np.array([left,right],dtype=np.int64)
    return res



@numba.njit
def laplace_smooth_multiple_haplotypes(posArr,valArr,scale): #this version does change the state of uncalled markers
    #posArr is still a 1D array
    #valArr is now multi-dimensional.  The rows correspond to haplotypes and the columns to positions

    maxDist = TruncatedLaplaceReach(scale/2)

    # print("scale is ",scale," and max distance is ",maxDist)
    
    left = 0
    focus = 0
    right = 0

    nHaplotypes = len(valArr)
    nSites = len(posArr)



    if nHaplotypes >= nSites:
        print("there are more haplotypes than sites in this dataset. check that you have passed the  position array and value array in the correct order")
        print("continuing evaluation as if everything were normal")
    #if nHaplotypes == nSites: raise Exception("number of haploptypes and sites are the same. Make sure valArr is a 2D array with haplotypes as rows and marker states as columns")
    res = np.zeros((nHaplotypes,nSites),dtype=np.int8)-1
    
    for focusIdx,focusPos in enumerate(posArr):
        # print(focusIdx)
        # if focusIdx>10:
        #     break

        
        while left<=focusIdx and (focusPos - posArr[left])>maxDist: left +=1
        while right< len(posArr)-1 and (posArr[right+1] - focusPos)<maxDist: right += 1
        weights = get_laplace_weights(posArr[left:right+1],focusIdx - left,scale)
        #print(weights)
        #print(left,focusIdx,right)
        # if len(valArr[0][left:right+1]) == 0:
        #     print('focus Idx = ', focusIdx, 'left = ', left, 'right = ', right)
        #     print(valArr[:,left:right+1])
        #     break
        for haplotypeIndex in range(nHaplotypes):
            
            #print(valArr[haplotypeIndex][left:right+1])
            # uncomment this  and comment out the other lines to make it keep all Us as Us
            # if valArr[haplotypeIndex][focusIdx] == 3: #if this haplotype specifically has an uncalled marker, don't change anything
            #     v = 3
            # else:
            #     v = get_laplace_smoothed_value(posArr[left:right+1],valArr[haplotypeIndex][left:right+1],focusIdx - left,weights)
            #print('val arr = ', valArr[haplotypeIndex][left:right+1])
            v = get_laplace_smoothed_value(posArr[left:right+1],valArr[haplotypeIndex][left:right+1],focusIdx - left,weights)
            #print(f'v = {v}')
            res[haplotypeIndex][focusIdx] = v
            
    return res        


@numba.njit(parallel=True)
def laplace_smooth_multiple_haplotypes_parallel(posArr, valArr, scale):
    """
    Parallel version of laplace_smooth_multiple_haplotypes using numba prange.
    Parallelizes across genomic sites for better performance.
    """
    maxDist = TruncatedLaplaceReach(scale/2)
    
    nHaplotypes = len(valArr)
    nSites = len(posArr)
    
    if nHaplotypes >= nSites:
        print("there are more haplotypes than sites in this dataset. check that you have passed the position array and value array in the correct order")
        print("continuing evaluation as if everything were normal")
    
    res = np.zeros((nHaplotypes, nSites), dtype=np.int8) - 1
    
    # Pre-compute intervals for all sites (this helps with parallelization)
    # IMPORTANT: Use scale/2 to match the sequential version
    intervals = get_interval_indices(posArr, scale/2)
    
    # Parallelize across sites using prange
    for focusIdx in prange(nSites):
        left = intervals[focusIdx][0]
        right = intervals[focusIdx][1]
        
        # Get weights for this site's window
        weights = get_laplace_weights(posArr[left:right+1], focusIdx - left, scale)
        
        # Process all haplotypes for this site
        for haplotypeIndex in range(nHaplotypes):
            v = get_laplace_smoothed_value(
                posArr[left:right+1], 
                valArr[haplotypeIndex][left:right+1], 
                focusIdx - left, 
                weights
            )
            res[haplotypeIndex][focusIdx] = v
    
    return res
