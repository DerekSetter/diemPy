#likely need to adjust what needs being imported, 
#for local imports, check which functions are being used and whether there are better functions defined elsewhere

import numpy as np
# np.set_printoptions(legacy = '1.25')

#import pandas as pd
#import csv
#import time
#import os
#import multiprocessing
# from collections import Counter
# import pickle

# from . import polarize as pol
# from . import smooth as ks

# left and right indices are inclusive. So interval is [l,r] inclusive of both ends.
def get_intervals(chrName, indName, statesList, posList, includeSingle = True):
    '''
    Function to get intervals for a given contig from states and positions. 
    single site intervals can be included or excluded. By default they are included.

    Args:

        chrName (str): Chromosome name.
        indName (str): Individual name.
        statesList (list): List of states at each site for this individual and chromosome.
        posList (list): List of positions.
        mapPosList (list): List of map positions.

    Returns:
        list: List of intervals.
    '''

    lidx = 0
    ridx = 0
    ivls = []
    while ridx <= len(statesList)-1:

        if ridx == len(statesList)-1:
            l = posList[lidx]
            r = posList[ridx]
            iv = Interval(chrName, indName, lidx, ridx, l, r, currentState)

            if includeSingle == True:
                ivls.append(iv)
            else:
                if r-l>0: 
                    ivls.append(iv)
            break

        currentState = statesList[lidx]
        if statesList[ridx+1] == currentState:
            ridx += 1
        else:
            l = posList[lidx]
            r = posList[ridx]

            iv = Interval(chrName, indName, lidx, ridx, l, r, currentState)

            if includeSingle == True:
                ivls.append(iv)
            else:
                if r-l>0: 
                    ivls.append(iv)

            ridx +=1
            lidx=ridx
            currentState = statesList[lidx]
            
    return ivls




# interval should also add info about (any) sites supporting the interval in addition to the left and right sites defining it
# this should be done in a way to look for gene conversion events later on
class Interval:
    '''
    Represents a genomic interval for a specific individual and chromosome.

    Args:

        chrName (str): Chromosome name.
        indName (str): Individual name.
        idxl (int): Left index (inclusive).
        idxr (int): Right index (inclusive). So slice of state matrix would be [idxl:idxr+1]
        l (float): Left position (physical).
        r (float): Right position (physical).
        state (int): State of the interval.

    :ivar str chrName: Chromosome name.
    :ivar str indName: Individual name.
    :ivar int idxl: Left index (inclusive).
    :ivar int idxr: Right index (inclusive). So slice of state matrix would be [idxl:idxr+1]
    :ivar float l: Left position (physical).
    :ivar float r: Right position (physical).
    :ivar int state: State of the interval.

    '''

    def __init__(self,chrName,indName,idxl,idxr,l,r,state):

        self.chrName = chrName
        self.indName = indName
        self.idxl = idxl
        self.idxr = idxr
        self.l = l
        self.r = r
        self.state = state



    def info(self):
        print(f"chr = {self.chrName}, ind = {self.indName}, idxl = {self.idxl}, idxr = {self.idxr}, l = {self.l}, r = {self.r}, state = {self.state}")

    def span(self):
        return self.r - self.l
    
    def mapSpan(self,chrLength):
        return (self.r - self.l)/chrLength # in on a 'unit map length' scale, i.e. 0 to 1
    

    
#the chromosome class contains the haplotype structure of a single chromosome.
# the individual and chromosome are indexedIt contains 
# for a single individual and single chromosome. Maybe 'individualChromsome' would be a better name?
# the individual is simply a list of the intervals (as defined above)
class Contig:
    
    '''
    Represents a contiguous sequence of genomic intervals for a specific individual and chromosome.
    
    Args:
        chrName (str): Chromosome name.
        indName (str): Individual name.
        intervalList (list): List of Interval objects.

    :ivar str chr: Chromosome name.
    :ivar str ind: Individual name.
    :ivar int num_intervals: Number of intervals.
    :ivar list intervals: List of Interval objects.

    '''

    
    
    # individual is a list of intervals pertaining to a single chromosome

    def __init__(self,chrName=None,indName=None,intervalList=None):
  
        self.intervals = intervalList
        if self.intervals is None:
            self.num_intervals = 0
        else:
            self.num_intervals = len(self.intervals)
        self.indName = indName
        self.chrName = chrName

        # self.getZeroIntervals()
        # self.getOneIntervals()
        # self.getTwoIntervals()
        # self.getThreeIntervals()
        
    

    def printIntervals(self,lim=10):
        print("formatting is as follows [leftPosition,rightPosition,state]")
        if lim is None:
            print([[x.l,x.r,x.state] for x in self.intervals]) 
        else:
            print([[x.l,x.r,x.state] for x in self.intervals[0:min(self.num_intervals,lim)]])


    def get_my_intervals_of_state(self,state):
        ivs = []
        for x in self.intervals:
            if x.state == state:
                ivs.append(x)
        return ivs


def pack_interval(interval):
    '''
    Packs an Interval object into a dictionary.
    Args:
        interval (Interval): Interval object to be packed.
    Returns:
        dict: Dictionary representation of the Interval object.
    '''
    return interval.__dict__

def unpack_interval(d):
    '''
    Unpacks a dictionary into an Interval object.
    Args:
        d (dict): Dictionary representation of an Interval object.
    Returns:
        Interval: Unpacked Interval object.
    '''
    blankArgs = [None for _ in range(7)]
    i = Interval(*blankArgs)
    i.__dict__.update(d)
    return i

def pack_intervalList(ivl):
    '''
    Packs a list of Interval objects into a list of dictionaries.
    Args:
        ivl (list): List of Interval objects to be packed.
    Returns:
        list: List of dictionary representations of the Interval objects.
    '''
    return [pack_interval(iv) for iv in ivl]

def unpack_intervalList(dlist):
    '''
    Unpacks a list of dictionaries into a list of Interval objects.
    Args:
        dlist (list): List of dictionary representations of Interval objects.
    Returns:
        list: List of unpacked Interval objects.
    '''
    return [unpack_interval(d) for d in dlist]

def pack_contig(contig):
    '''
    Packs a Contig object into a dictionary. This requires 'packing' the list of Interval objects as well.
    Args:
        contig (Contig): Contig object to be packed.
    Returns:
        dict: Dictionary representation of the Contig object.
    '''
    d = contig.__dict__.copy()
    d['intervals'] = pack_intervalList(contig.intervals)
    return d

def unpack_contig(d):
    contig = Contig()

    contig.__dict__.update(d)
    contig.intervals = unpack_intervalList(d['intervals'])
    return contig

def pack_contig_matrix(cArr):
    '''
    Packs a Matrix of Contig objects into a matrix of dictionaries.
    Matrix is (num_chromosomes, num_individuals) in sort order of the diemtype parent object.

    Args:
        cArr (np.array dtype=object): Matrix of Contig objects to be packed.
    Returns:
        list: List of dictionary representations of the Contig objects.
    '''
    
    return [[pack_contig(c) for c in row] for row in cArr]

def unpack_contig_matrix(dArr):
    '''
    Unpacks a Matrix of dictionaries into a matrix of Contig objects.
    Matrix is (num_chromosomes, num_individuals) in sort order of the diemtype parent object.

    Args:
        dArr (list): List of dictionary representations of Contig objects.
    Returns:
        np.array dtype=object: Matrix of unpacked Contig objects.
    '''
    return np.array([[unpack_contig(d) for d in row] for row in dArr], dtype=object)






def build_contig_matrix(diemType,includeSingle = True):
    '''
    Creates a matrix of Contig objects from a DiemType object.
    Matrix is (num_chromosomes, num_individuals) in sort order of the diemtype parent object.

    Args:
        diemType (DiemType): DiemType object from which to create the Contig matrix.
    Returns:
        np.array dtype=object: Matrix of Contig objects.
    '''
    nChrs = len(diemType.chrNames)
    nInds = len(diemType.indNames)

    cArr = np.empty((nChrs, nInds), dtype=object)

    for cIdx in range(nChrs):
        chrName = diemType.chrNames[cIdx]
        for indIdx in range(nInds):
            indName = diemType.indNames[indIdx]
            statesList = diemType.DMBC[cIdx][indIdx]
            posList = diemType.posByChr[cIdx]

            ivl = get_intervals(chrName, indName, statesList, posList, includeSingle=includeSingle)
            contig = Contig(chrName, indName, ivl)
            cArr[cIdx, indIdx] = contig

    return cArr