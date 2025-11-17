from . import polarize as pol 
from . import smooth as ks
from . import contigs as ct
import numpy as np
import copy
import pandas as pd
import pickle

# useful function for flipping the polarity of a diem matrix.
# needed so that we can read in the 'polarized' output of diem
# which stores a polarity but not the flipped matrix
# so we can just flip the matrix by the polarity on import
# can also be used to reconstruct original polarity by passing newPolarity = 0 for all markers
# and oldPolarity = the polarity stored by diem
def flip_polarity(diemMatrix,newPolarity,oldPolarity=None):
    '''
    Flip the polarity of the Diem Matrix according to the polarity array.
    diem matrix is an array with inds as rows and sites as columns
    polarity is a numpy array of shape nMarkers, with entries 0 or 1, where 1 indicates that the marker should be flipped.

    Args:
        diemMatrix (np.ndarray): The Diem Matrix to be flipped.
        newPolarity (np.ndarray): Array indicating the desired polarity for each marker (0 or 1).
        oldPolarity (np.ndarray, optional): Array indicating the current polarity for each marker (0 or 1). If oldPolarity = None, it flips the markers where newPolarity is 1.
    Returns:
        np.ndarray: The Diem Matrix with updated polarity.
    '''
    if oldPolarity is None:
        oldPolarity = np.zeros(len(newPolarity), dtype=np.int8)
    
    cols_to_flip = np.where(newPolarity != oldPolarity)[0]
    if len(cols_to_flip) == 0:
        return diemMatrix  # No changes needed

    #print('number of columns to flip:', len(cols_to_flip))
    
    arr = copy.deepcopy(diemMatrix)
    # Use a temporary value to avoid overwriting
    for col in cols_to_flip:
        col_data = arr[:, col]
        col_data[col_data == 1] = -1
        col_data[col_data == 3] = 1
        col_data[col_data == -1] = 3
    return arr
    
    # this for some reason did not work. it did not modify the array
    # but it should be faster than looping as above

    # arr[:, cols_to_flip][arr[:, cols_to_flip] == 1] = -1
    # arr[:, cols_to_flip][arr[:, cols_to_flip] == 3] = 1
    # arr[:, cols_to_flip][arr[:, cols_to_flip] == -1] = 3
    # return arr

# here, SM (State Matrix) is equivalent to DM (Diem Matrix) as it is now called
def A4_from_stateMatrix(SM,ploidies):
    I4 = np.zeros((len(SM),4))
    for idx,individual in enumerate(SM):
        numZeros = np.count_nonzero(individual==0)
        numOnes = np.count_nonzero(individual==1)
        numTwos = np.count_nonzero(individual==2)
        numThrees = np.count_nonzero(individual==3)
        I4[idx][:] = np.array([numZeros,numOnes,numTwos,numThrees])
    A4 = np.dot(np.diag(ploidies),I4)
    return A4

# defined as get_hybrid_index(A4) in diem polarization code, copied for use here
def get_hybrid_index(A4):
    #A4 is the matrix of (nHaps x states) with entries being the ploidy-adjusted counts
    nHaps = len(A4)
    HIarr = np.zeros(nHaps)
    for idx,counts in enumerate(A4):
        hiNum = counts[1]*0 + counts[2]*1 + counts[3]*2
        hiDenom = 2*(counts[1] + counts[2] + counts[3])
        hi = hiNum/hiDenom
        HIarr[idx] = hi
    return HIarr

def get_resort_order(inds1,inds2):
    """
    Get the order to resort inds2 to  match inds1
    
    Args:
        inds1 (np.ndarray): the array with the reference ordering
        inds2 (np.ndarray): the array to be reordered
    """

    newOrder = [np.argwhere(inds2 == s)[0][0] for s in inds1]
    newOrder = np.array(newOrder)

    return newOrder



# possible additions to the diemtype class:
#   re-polarize function:  takes an alternate polarity and changes state matrix. could 'de-polarize' data, as the input data is always '0' polarity
class DiemType:
    """
    Class describing the raw data for state matrices, and functions for thresholding, kernel smoothing, etc.
    
    Args:
        DMBC (List[np.ndarray]): State matrix by chromosome. For arrays, each row is an individual, each column a marker.
        indNames (np.ndarray): Names of individuals, same order as DMBC.
        chrPloidies (List[np.ndarray]): for each chromosome, the Ploidy of each individual, same order as DMBC.
        chrNames (np.ndarray): Names of chromosomes as ordered in DMBC.
        posByChr (List[np.ndarray]): Positions of markers by chromosome. 
        chrLengths (List[int]): Lengths of chromosomes. 
        exclusionsByChr (List[np.ndarray], optional): List of arrays of positions indicating which sites to exclude for each chromosome when polarizing. If None, includes all sites. If some sites are excluded, but a given chromosome has no exclusions, that list entry should be None.
        indExclusions (np.ndarray, optional): array of names of individuals to exclude when polarizing. If None, includes all individuals.

    :ivar DMBC: List[np.ndarray]. State matrix by chromosome.
    :ivar indNames: np.ndarray. Names of individuals.
    :ivar chrPloidies: List[np.ndarray]. For each chromosome, the ploidy of each individual.
    :ivar chrNames: np.ndarray. Names of chromosomes.
    :ivar posByChr: List[np.ndarray]. Positions of markers by chromosome.
    :ivar chrLengths: List[int]. Lengths of chromosomes.
    :ivar MapBC: List[np.ndarray]. Genetic map positions by chromosome, computed on initialization.
    :ivar HIs: np.ndarray. Heterozygosity indices, to be computed.
    :ivar PolByChr: List[np.ndarray]. Polymorphism matrix by chromosome, to be computed.
    :ivar initialPolByChr: List[np.ndarray]. Initial polarity by chromosome for EM start (random or test), to be computed.
    :ivar DIByChr: List[np.ndarray]. Diagnostic index by chromosome, to be computed.
    :ivar SupportByChr: List[np.ndarray]. Support values by chromosome, to be computed.
    :ivar threshold: float. Threshold value to be set.
    :ivar smoothScale: float. Scale for kernel smoothing, to be set.
    :ivar contigMatrix: np.ndarray dtype=object. Matrix of Contig objects, to be created.
    """

    def __init__(self,DMBC, indNames, chrPloidies, chrNames, posByChr,chrLengths,exclusionsByChr=None,indExclusions=None):
            
        self.DMBC = DMBC
        self.indNames = indNames
        self.chrPloidies = chrPloidies
        self.chrNames = chrNames
        self.posByChr = posByChr
        self.chrLengths = chrLengths

        self.initMaplength()

        self.HIs = None
        self.initialPolByChr = None
        self.PolByChr = None
        self.DIByChr = None
        self.SupportByChr = None
        self.threshold = None
        self.smoothScale = None
        

        self.contigMatrix = None

        self.siteExclusionsByChr = exclusionsByChr
        self.indExclusions = indExclusions
        #self.idealMarker = None
        #self.barrierBreakIdx = None

    def initMaplength(self):
        if self.posByChr is None:
            self.MapBC = None
            return None
        self.MapBC = []
        for idx in range(len(self.chrNames)):
            mapLength = self.posByChr[idx]
            mapLength = mapLength/self.chrLengths[idx]
            self.MapBC.append(mapLength)

    def add_individual_exclusions(self,filePath):
        df = pd.read_csv(filePath,header=None)
        self.indExclusions = np.array(df.iloc[:,0].tolist())

    def add_site_exclusions(self,filePath):
        self.siteExclusionsByChr = [None]*len(self.chrNames)
        skipped_chromosomes = set()
        
        with open(filePath,'r') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:  # Skip empty lines
                    continue
                    
                chrName, start, end = clean_line.split('\t')
                start = int(start) + 1
                end = int(end) + 1

                # Check if chromosome exists in dataset
                chr_matches = np.where(self.chrNames == chrName)[0]
                if len(chr_matches) == 0:
                    skipped_chromosomes.add(chrName)
                    continue
                    
                chrIdx = chr_matches[0]
                
                if self.siteExclusionsByChr[chrIdx] is None:
                    self.siteExclusionsByChr[chrIdx] = []
                
                # Append the indices of the positions in self.posByChr[chrIdx] that are >= start and < end
                indices = np.where((self.posByChr[chrIdx] >= start) & (self.posByChr[chrIdx] < end))[0]
                self.siteExclusionsByChr[chrIdx] = self.siteExclusionsByChr[chrIdx] + indices.tolist()
        
        # Report any skipped chromosomes
        if skipped_chromosomes:
            print(f"Warning: The following chromosomes in the exclusions file were not found in the dataset and were skipped: {sorted(skipped_chromosomes)}")
        

    def computeHIs(self):
        """
        Compute heterozygosity indices for each individual.
        """
        if self.PolByChr is None:
            print("data must be polarized before computing HIs. Resorting will be done automatically after polarizing, thresholding, or smoothing.")
            return None    
        else:

            A4List = []
            for idx, chr in enumerate(self.DMBC):
                ploidies = self.chrPloidies[idx]
                A4 = A4_from_stateMatrix(chr, ploidies)
                A4List.append(A4)

            A4Total = np.sum(A4List, axis=0)
            newHIs = get_hybrid_index(A4Total)

        return newHIs
            
    # this function needs to be tested with respect to contig matrix resort
    def sort(self,newHIs = None):
        """
        Sort DMBC and individuals (and their ploidies) by hybrid index.
        """
        if self.PolByChr is None:
            print("data must be polarized before sorting by HI.")
            print("Note: resorting is NOT automatic after polarizing, thresholding, or smoothing")
            print("to resort data, use the method self.sort()")
            return None
        else:
            oldOrder = np.arange(len(self.indNames))
            newHIs = self.computeHIs()    
            newOrder = np.argsort(newHIs)

            if np.all(self.HIs == newHIs) and np.all(oldOrder == newOrder):
                print("HIs already up to date, and no sorting needed")
            elif np.all(oldOrder == newOrder):
                print("HIs computed and new values updated, but ordering did not change")
            else:
                for idx,arr in enumerate(self.DMBC):
                    self.DMBC[idx] = arr[newOrder,:]
                    self.chrPloidies[idx] = self.chrPloidies[idx][newOrder]
                self.indNames = self.indNames[newOrder]
                self.HIs = newHIs[newOrder]

                if self.contigMatrix is not None:
                    for idx, arr in enumerate(self.contigMatrix):
                        self.contigMatrix[idx] = arr[newOrder,:]
                print("new HIs computer and individuals resorted by HI")

    # this function needs to be tested with respect to contig matrix resort
    def sort_as(self,d2):
        '''
        Sort the current instance (self) by the order (of individuals) of another instance (d2).
        
        Args:
            d2 (DiemType): the instance to be used for sorting
        '''
        inds1 = self.indNames
        inds2 = d2.indNames

        if np.array_equal(inds1,inds2):
            print("the datasets are already in the same order")
            return None
        
        sameSet = np.array_equal(np.sort(inds1), np.sort(inds2))
        if not sameSet:
            print("the datasets do not contain the same individuals")
            return None
        else:
            newOrder = get_resort_order(inds2,inds1)
            print(newOrder)
            for idx,arr in enumerate(self.DMBC):
                self.DMBC[idx] = arr[newOrder,:]
                self.chrPloidies[idx] = self.chrPloidies[idx][newOrder]
            self.indNames = self.indNames[newOrder]
            self.HIs = self.HIs[newOrder]


            if self.contigMatrix is not None:
                for idx, arr in enumerate(self.contigMatrix):
                    self.contigMatrix[idx] = arr[newOrder,:]

            print("dataset has been re-ordered")
            return None

    # this function needs to be tested with respect to contig matrix resort
    def sort_by(self,newOrder):
        '''
        Sort the current instance (self) by a given ordering of individuals.
        
        Args:
            newOrder (List[int] or np.ndarray): the new order of individuals, given as indices.
        '''
        # if newOrder is a list, convert to np.ndarray
        if isinstance(newOrder, list):
            newOrder = np.array(newOrder)
        # check that newOrder is a np.ndarray
        elif not isinstance(newOrder, np.ndarray):
            print("newOrder must be a list or np.ndarray")
            return None

        if len(newOrder) != len(self.indNames):
            print("the new order does not have the same length as the number of individuals")
            return None
        elif set(newOrder) != set(range(len(self.indNames))):
            print("the new order does not contain the same indices as the number of individuals")
            return None
        else:
            for idx,arr in enumerate(self.DMBC):
                self.DMBC[idx] = arr[newOrder,:]
                self.chrPloidies[idx] = self.chrPloidies[idx][newOrder]
            self.indNames = self.indNames[newOrder]
            self.HIs = self.HIs[newOrder]
            if self.contigMatrix is not None:
                for idx, arr in enumerate(self.contigMatrix):
                    self.contigMatrix[idx] = arr[newOrder,:]
            print("dataset has been re-ordered")
            return None

    def copy(self):
        '''
        Create a deep copy of the current instance.
        '''
        return copy.deepcopy(self)
    


    #note that these optional arguments are currently defined in the run_em_linear and run_em_parallel functions
    #meaning they are actually over-ridden here. Not a huge issue but could be refactored later if desired 
    def polarize(self,ncores=1,boolTestData=False,maxItt=500,epsilon=0.99999,sort_by_HI=False):

        """
        Polarize the state matrices by initializing test polarities and running the EM algorithm. Does not change self, but rather returns a polarized copy. Note that it will use the individual and site exclusions defined in self. 

        Args:
            ncores (int): number of cores to use for parallel processing. If 1, runs in serial.
            boolTestData (bool): if True, initializes polarity using test data method. If False, initializes polarity randomly.
            maxItt (int, optional): Maximum number of iterations for the EM algorithm. Default is 500.
            epsilon (float, optional): Convergence threshold for the EM algorithm. Default is 0.99

        Returns:
            DiemType: A new DiemType instance with polarized data.
        """

        indExcludedIndices = None

        if self.indExclusions is not None:
            indExcludedIndices = np.where(np.isin(self.indNames,self.indExclusions))[0]


        print("convert state matrix to Marray")
        initMBC = [pol.stateMatrix_to_MArray(x) for x in self.DMBC]
        initPolBC = []

        if boolTestData == True:
            print("initializing test polarity")
            for idx,M in enumerate(initMBC):
                thisPol,thisM = pol.initialize_test_polarity(M)
                initMBC[idx] = thisM
                initPolBC.append(thisPol)
        else:
            print("initializing random polarity")
            for idx,M in enumerate(initMBC):
                thisPol,thisM = pol.initialize_random_polarity(M)
                initMBC[idx] = thisM
                initPolBC.append(thisPol)
       
        if ncores == 1:
            print("running EM algorithm on a single core using diem_linear")
            MBC_out, polBC_out, DIBC_out,SupportBC_out = pol.run_em_linear(initMBC, initPolBC, self.chrPloidies,sitesExcludedByChr=self.siteExclusionsByChr, individualsExcluded=indExcludedIndices,maxItt=maxItt,epsilon=epsilon)
        else:
            print("running EM algorithm in parallel using diem_parallel with ",ncores," cores")
            MBC_out, polBC_out, DIBC_out,SupportBC_out = pol.run_em_parallel(initMBC, initPolBC, self.chrPloidies,sitesExcludedByChr=self.siteExclusionsByChr,individualsExcluded=indExcludedIndices,maxItt=maxItt,epsilon=epsilon,nCPUs=ncores)

        print("updating polarizations, DIs, Supports, initialPolarity,and state matrices")

        a = copy.deepcopy(self)
        #a = self  #this will modify the original instance, not what we want
        
        a.initialPolByChr = [x for x in initPolBC] # ensure this is a copy, not reference.
        a.PolByChr = polBC_out
        a.DIByChr = DIBC_out
        a.SupportByChr = SupportBC_out
        a.DMBC = [pol.MArray_to_stateMatrix(x) for x in MBC_out]
        
        a.HIs = a.computeHIs()
        if sort_by_HI == True:
            print("re-sorting individuals by HI")
            a.sort()

        return a
    

    def apply_threshold(self, threshold: float,sort_by_HI=False):
        """
        Apply a threshold to the diagnostic indices and update to remove sites below threshold di. Returns a copy of the modified instance.

        Args:
            threshold (float): Threshold value for diagnostic index. Sites with DI below this value will be removed.

        Returns:
            DiemType: A new DiemType instance with sites below the threshold removed.
        """

        a =copy.deepcopy(self)
        proportionRetained = []
        a.threshold = threshold
        for idx in range(len(a.DMBC)):
            myFilter = a.DIByChr[idx]>=threshold
            
            proportionRetained.append(sum(myFilter)/len(myFilter))
            trans = a.DMBC[idx].transpose()
            trans = trans[myFilter]
            a.DMBC[idx] = trans.transpose()
            a.posByChr[idx] = a.posByChr[idx][myFilter]

            a.PolByChr[idx] = a.PolByChr[idx][myFilter]
            a.DIByChr[idx] = a.DIByChr[idx][myFilter]
            a.MapBC[idx] = a.MapBC[idx][myFilter]
        print("proportion of each chromosome retained after thresholding:")
        print(proportionRetained)

        print("thresholding done")
        if sort_by_HI == True:
            a.sort()
            print("new hybrid indices computed and individuals have been resorted by HI")
        else:
            print("hybrid indices have likely changed, but have not been updated")
            print("to update HIs without re-sorting, update the attribute a.HIs = a.computeHIs()")
            print("You may also call the sort() method on the resulting data if you wish to both re-compute and resort")
        
        return a
    
    def smooth(self,scale: float ,reSort=False,reSmooth=False,parallel=True):
        """
        Smooth and return a copy of the state matrices using a Laplace kernel . defaults to NOT resorting by hybrid index. This allows for direct comparison to pre-smoothed data.  May later resort using self.sort() on resulting data.

        Args:
            scale (float): Scale parameter for the Laplace kernel smoothing.
            reSort (bool, optional): If True, resorts individuals by hybrid index after smoothing. Default is False.
            reSmooth (bool, optional): If True, allows re-smoothing even if smoothing has already been done. Default is False.
            parallel (bool, optional): If True, uses parallel processing for smoothing. Default is True.

        Returns:
            DiemType: A new DiemType instance with smoothed state matrices.
        """

        if self.smoothScale is not None:
            print("smoothing has already been done with scale ",self.smoothScale)
            if reSmooth == False:
                print("In order to re-smooth the data, please set reSmooth=True. Note this will over-write the previously-used smoothing scale value")
                return None
            else:
                print("re-smoothing the data with new scale ",scale)
        else:
            print("smoothing the data with scale ",scale)

        a = copy.deepcopy(self)
        a.smoothScale = scale

        if parallel == True:
            print("using parallel smoothing")
        else:
            print("using serial smoothing")
            
        for idx in range(len(a.DMBC)):
            if parallel == False:
                thisStateMatSmoothed = ks.laplace_smooth_multiple_haplotypes(a.MapBC[idx],a.DMBC[idx],scale)
            else:
                thisStateMatSmoothed = ks.laplace_smooth_multiple_haplotypes_parallel(a.MapBC[idx],a.DMBC[idx],scale)

            a.DMBC[idx] = thisStateMatSmoothed
        if reSort == False:
            print("smoothing done, but not resorting by HI. You may call the sort() method on the resulting data if you wish to resort")
            return a
        else:
            print("smoothing done and data has been resorted by HI")
            a.sort()
            return a
        
    def create_contig_matrix(self,includeSingle = True):
        """
        Create a matrix of Contig objects from the current DiemType instance and store it in self.contigMatrix.

        Args:
            includeSingle (bool, optional): If True, includes contigs with a single marker. Default is True.
        """
        self.contigMatrix = ct.build_contig_matrix(self,includeSingle=includeSingle)
        print("contig matrix created and stored in self.contigMatrix")
        return None
    
    def get_intervals_of_state(self,state,individualSubset = None,chromosomeSubset=None):
        """
        Get intervals of a specified state for given individuals and chromosomes.

        Args:
            state (int): The state to find intervals for (0, 1, 2, or 3).
            individualSubset (List[int], optional): List of individual indices to include. If None, includes all individuals.
            chromosomeSubset (List[int], optional): List of chromosome indices to include. If None, includes all chromosomes.

        Returns:
            List[Interval]: A list of Interval objects for the specified state across the specified individuals and chromosomes.
        """
        if self.contigMatrix is None:
            print("contig matrix has not been created. Please run the create_contig_matrix() method first.")
            return None
        if state not in [0,1,2,3]:
            print("state must be 0, 1, 2, or 3")
            return None
        intervals = []
        for idxChr in range(len(self.chrNames)):
            if chromosomeSubset is not None and idxChr not in chromosomeSubset:
                continue
            for idxInd in range(len(self.indNames)):
                if individualSubset is not None and idxInd not in individualSubset:
                    continue
                thisContig = self.contigMatrix[idxChr, idxInd]
                chrIntervals =thisContig.get_my_intervals_of_state(state)
                intervals.extend(chrIntervals)
        return intervals

    def intervals_to_bed(self,outputDir):
        """
        Export intervals of each state to BED files for each chromosome and state.

        Args:
            outputDir (str): Directory to save the BED files.
        """
        if self.contigMatrix is None:
            print("contig matrix has not been created. Please run the create_contig_matrix() method first.")
            return None
        ct.export_contigs_to_ind_bed_files(self,outputDir)
        print("BED files created in directory: ",outputDir)
        return None


# save diemtype function needs to be updated.  The dictionary of variables is saved exactly as below
# however, now the contigMatrix needs to be 'packed' as well in order to be saved correctly.

def save_DiemType(diemTypeObj, pcklPath):
    '''
    Save the dictionary of variables for a DiemType object to a pickle file.
    File extensions something like my_name.diemtype.dict.pkl

    Args:
        diemTypeObj (DiemType): The DiemType object to be saved.
        pcklPath (str): Path to save the pickle file.
    '''
    # the contigMatrix needs to be packaged
    # however, we do not want to overwrite the original contigMatrix
    # so we pack the dictionary version of the contigMatrix
    # and then save that

    d = diemTypeObj.__dict__.copy()
    if d['contigMatrix'] is not None:
        d['contigMatrix'] = ct.pack_contig_matrix(d['contigMatrix'])

    with open(pcklPath, 'wb') as f:
        pickle.dump(d, f,protocol=pickle.HIGHEST_PROTOCOL)


# as in saveDiemType, the contigMatrix needs to be unpacked after loading
def load_DiemType(pcklPath):
    '''
    Load a dict of objects that make a DiemType, then construct that DiemType from the loaded data.
    This helps ensure that, so long as variables are not renamed or removed, DiemType objects can be saved and loaded even if the class definition changes.
    attributes can be added, and we could later add code to handle removed attributes if needed.


    Args:
        pcklPath (str): Path to the pickle file.

    Returns:
        DiemType: The loaded DiemType object.
    '''
    blankArgs = [None for x in range(6)]
    d = DiemType(*blankArgs)

    with open(pcklPath, 'rb') as f:
        d.__dict__.update(pickle.load(f))

    if d.contigMatrix is not None:
        d.contigMatrix = ct.unpack_contig_matrix(d.contigMatrix)

    return d