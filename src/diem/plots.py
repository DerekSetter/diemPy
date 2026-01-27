from __future__ import annotations


# ---- stdlib ----
import copy
from collections import defaultdict, Counter
from itertools import groupby, chain


# ---- numpy / pandas ----
import numpy as np
import pandas as pd


# ---- matplotlib ----
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider


# ---- scipy ----
from scipy.interpolate import interp1d


# ---- diem internals ----
#from . import polarize as pol
from . import smooth
from . import contigs as ct


# explicitly used smoothing entry point
from .smooth import laplace_smooth_multiple_haplotypes
from fractions import Fraction



""" stuff from Nina's other files STARTING """
##############################
#### Mathematica2Python
### Author: Stuart J.E. Baird
###############################

#import itertools
#import multiprocessing as mp
#import numpy as np


def Split(seq, same_test=lambda a, b: a == b): # code credit ChatGPT 5.2 (asked for Mca equiv)
    if not seq:
        return []
    out = [[seq[0]]]
    for x in seq[1:]:
        (out[-1] if same_test(out[-1][-1], x) else out.append([x])) and out[-1].append(x)
    return out


def SplitBy(lst, test):  # code credit: BPL @ StackOverflow
    return Split(lst, lambda x, y: test(x) == test(y))



def RichRLE(lst):
    slst = Split(lst, lambda x, y: y == x)
    cumstart = 0
    states = []
    lengths = []
    starts = []
    ends = []
    for i in slst:
        leni = len(i)
        states.append(i[0])
        lengths.append(leni)
        starts.append(cumstart)
        cumstart += leni
        ends.append(cumstart - 1)
    return [states, lengths, starts, ends]


def Map(f, lst): return list(map(f, lst))


def ParallelMap(f, lst):
    pool = mp.Pool()
    return list(pool.map(f, lst))


def Flatten(lstOlists): return list(chain.from_iterable(lstOlists)) #itertools


def StringJoin(slst):
    separator = ''
    return separator.join(slst)


def Transpose(mat): return list(np.array(mat).T)  # care here - hidden type casting on heterogeneous 'mat'rices

def StringTranspose(slst): return Map(StringJoin, Transpose(Map(Characters, slst)))


def Tally(lst):  # single pass so in principle fast ( O(n) ) but answers unsorted
    states = []
    tally = []
    for x in lst:
        p = FirstPosition(states, x)
        if p == []:
            states.append(x)
            tally.append([x, 1])
        else:
            tally[p[0]][1] += 1
    return tally

def Second(lst): return lst[1]

def Total(lst): return sum(lst)

def Join(lst1, lst2): return lst1 + lst2


def Take(lst, n):
    if n > 0:
        ans = lst[:n]
    elif n == 0:
        ans = lst
    else:
        ans = lst[n:]
    return ans


def Drop(lst, n):
    if n > 0:
        ans = lst[n:]
    elif n == 0:
        ans = lst
    else:
        ans = lst[:n]
    return ans

def FirstPosition(lst, elem):
    i = -1
    pos = []
    for l in lst:
        i += 1
        if l == elem:
            pos.append(i)
            break
    return pos

def Characters(s): return [*s]

def StringTakeList(string, lengths):
    substrings = []
    current_index = 0
    for length in lengths:
        substrings.append(string[current_index:current_index + length])
        current_index += length
    return substrings

""" Endof Mca2python """

##############################
#### From DIEMPy
### Author: Stuart J.E. Baird
###############################

#import numpy as np
#from collections import Counter

#from .mathematica2python import *


StringReplace20_dict = str.maketrans('02', '20')

def StringReplace20(text):
    """will _!simultaneously!_ replace 2->0 and 0->2"""
    return text.translate(StringReplace20_dict)


def sStateCount(s):
    counts = Map(Second, Tally(Join(["0", "1", "2"], Characters(s))))
    nU = Total(
        Drop(counts, 3)
    )  # only the three 'call' chars above are not U encodings!
    counts = list(np.array(Take(counts, 3)) - 1)
    return Join([nU], counts)


def csStateCount(cs):
    """
    This function counts diem chars; input cs is char list or string; output is an np.array
    """
    ans = Counter("_012")
    ans.update(cs)
    return np.array(list(ans.values())) - 1


def pHetErrOnString(s):
    sCount = sStateCount(s)
    callTotal = Total(Drop(sCount, 1))
    if callTotal > 0:
        ans = (
            Total(np.array(sCount) * [0, 0, 1, 2]) / (2 * callTotal),
            sCount[2] / callTotal,
            sCount[0] / Total(sCount),
        )
    else:  # no calls... are there any Us?
        if sCount[0] > 0:
            pErr = 1
        else:
            pErr = "NA"
        ans = ("NA", "NA", pErr)
    return ans





""" stuff from Nina's other files ENDING """
""" new support by SJEB STARTING"""
"""
def genomes_summary_given_DI(aDT, DIthreshold: float):
    #Summarise genomes of each individual. Modelled on diemType:compute_HIs


    # modified from A4_from_stateMatrix(SM,ploidies) in Derek's diem polarization code, copied for use here
    # here, SM (State Matrix) is equivalent to DM (Diem Matrix) as it is now called
    def DIfiltered_A4_from_stateMatrix(SM,ploidies,DIfilter):
        I4 = np.zeros((len(SM),4))
        for idx,individual in enumerate(SM):
            numZeros = np.count_nonzero(individual[DIfilter]==0)
            numOnes = np.count_nonzero(individual[DIfilter]==1)
            numTwos = np.count_nonzero(individual[DIfilter]==2)
            numThrees = np.count_nonzero(individual[DIfilter]==3)
            I4[idx][:] = np.array([numZeros,numOnes,numTwos,numThrees])
        A4 = np.dot(np.diag(ploidies),I4)
        return A4

    # modified from get_hybrid_index(A4) in Derek's diem polarization code, copied for use here
    def get_frequencies(A4):
        #A4 is the matrix of (nHaps x states) with entries being the ploidy-adjusted counts
        nInds = len(A4)
        HIarr = np.zeros(nInds)
        HOM1arr = np.zeros(nInds)
        HETarr = np.zeros(nInds)
        HOM2arr = np.zeros(nInds)
        Uarr = np.zeros(nInds)
        for idx,counts in enumerate(A4):
            hiNum = counts[1]*0 + counts[2]*1 + counts[3]*2
            stateDenom = counts[1] + counts[2] + counts[3] + counts[0]
            dipDenom = counts[1] + counts[2] + counts[3]
            hapDenom = 2*dipDenom
            hi = hiNum/hapDenom
            HIarr[idx] = hi
            HOM1arr[idx] = counts[1]/dipDenom
            HETarr[idx] = counts[2]/dipDenom
            HOM2arr[idx] = counts[3]/dipDenom
            Uarr[idx] = counts[0]/stateDenom
        return [HIarr,HOM1arr,HETarr,HOM2arr,Uarr]
    
    A4List = []
    RetainedNumer = []
    RetainedDenom = []
    for idx, chr in enumerate(aDT.DMBC):
        ploidies = aDT.chrPloidies[idx]
        
        DIfilter = aDT.DIByChr[idx]>=DIthreshold
        RetainedNumer.append(sum(DIfilter))
        RetainedDenom.append(len(DIfilter))
 
        A4 = DIfiltered_A4_from_stateMatrix(chr, ploidies, DIfilter)
        A4List.append(A4)

    A4Total = np.sum(A4List, axis=0)
    summaries = get_frequencies(A4Total)

    return [summaries,sum(RetainedNumer),sum(RetainedDenom)]
"""

def read_diem_bed_4_plots(bed_file_path, meta_file_path):
    """
    This is completely ripped from Derek's read_diem_bed!!!
    It returns a pandas dataframe of the bed_file, POLARISED (if hasPolarity)

    Derek comments:
    Fast version of read_diem_bed with significant performance improvements.
    
    Parameters:
    bed_file_path (str): Path to the diem BED file.
    meta_file_path (str): Path to the diem metadata file.

    Returns:
    DiemType: DiemType object containing the diem BED data.
    """
    
    # Read metadata - no changes needed here as it's already fast
    df_meta = pd.read_csv(meta_file_path, sep='\t')
    chrNames = np.array(df_meta['#Chrom'].values)
    chrLengths = np.array(df_meta['RefEnd0'].values) - np.array(df_meta['RefStart0'].values)
    sampleNames = np.array(df_meta.columns[6:])
    
    ploidyByChr = []
    for chr in chrNames:
        row = df_meta[df_meta['#Chrom'] == chr]
        ploidy = np.array(row.iloc[0,6:].values, dtype=int)
        ploidyByChr.append(ploidy)
    
    # Fast preamble reading - same as before
    preamble = []
    nSkipLines = 0
    individualsMasked = None
    with open(bed_file_path, 'r') as f:
        for line in f:
            if line.startswith('##'):
                preamble.append(line.strip())
                if line.startswith('##IndividualsMasked='):
                    clean_line = line.strip().removeprefix('##IndividualsMasked=')
                    if clean_line == 'None':
                        individualsMasked = None
                    else:
                        individualsMasked = clean_line.split(',')
                nSkipLines += 1
            else:
                break
    
    # Determine column names
    if len(preamble) > 0:
        hasPolarity = True
        column_names = [
            'chrom', 'start', 'end', 'qual', 'ref', 
            'SeqAlleles', 'SNV', 'nVNTs', 
            'exclusion_criterion', 'diem_genotype','nullPolarity','polarity',
            'DI','Support','masked'
        ]
    else:
        hasPolarity = False
        column_names = [
            'chrom', 'start', 'end', 'qual', 'ref', 
            'SeqAlleles', 'SNV', 'nVNTs', 
            'exclusion_criterion', 'diem_genotype'
        ]
    
    # Read the entire BED file at once
    df_bed = pd.read_csv(bed_file_path, sep='\t', names=column_names, skiprows=nSkipLines+1)

    # Polarise - these last lines SJEB
                
    if hasPolarity:
        print('updating genotype polarities')
        mask = df_bed['polarity'] == 1
        df_bed.loc[mask, 'diem_genotype'] = df_bed.loc[mask, 'diem_genotype'].apply(StringReplace20)
        
    return df_bed,sampleNames

# ChatGPT 5.2 speed optimised version
def genomes_summary_given_DI(aDT, DIthreshold: float):
    """
    Vectorized, faster version.
    """

    nInds = aDT.DMBC[0].shape[0]
    A4Total = np.zeros((nInds, 4), dtype=float)

    RetainedNumer = 0
    RetainedDenom = 0

    for idx, SM in enumerate(aDT.DMBC):
        ploidies = aDT.chrPloidies[idx]
        DIfilter = aDT.DIByChr[idx] >= DIthreshold

        RetainedNumer += np.count_nonzero(DIfilter)
        RetainedDenom += DIfilter.size

        # Apply DI filter once
        SMf = SM[:, DIfilter]

        # Vectorized state counts
        I4 = np.stack([
            np.count_nonzero(SMf == 0, axis=1),
            np.count_nonzero(SMf == 1, axis=1),
            np.count_nonzero(SMf == 2, axis=1),
            np.count_nonzero(SMf == 3, axis=1),
        ], axis=1)

        # Apply ploidy weights and accumulate
        A4Total += ploidies[:, None] * I4

    # ---- vectorized frequency computation ----

    counts0 = A4Total[:, 0]
    counts1 = A4Total[:, 1]
    counts2 = A4Total[:, 2]
    counts3 = A4Total[:, 3]

    dipDenom = counts1 + counts2 + counts3
    hapDenom = 2 * dipDenom
    stateDenom = dipDenom + counts0

    HI = (counts2 + 2 * counts3) / hapDenom
    HOM1 = counts1 / dipDenom
    HET = counts2 / dipDenom
    HOM2 = counts3 / dipDenom
    U = counts0 / stateDenom

    summaries = [HI, HOM1, HET, HOM2, U]

    return summaries, RetainedNumer, RetainedDenom

def genomes_summary_given_DI_by_chromosome(aDT, DIthreshold, chrom_idx):
    """
    Per-chromosome version of genomes_summary_given_DI.
    Returns per-individual summaries for ONE chromosome.
    """

    SM = aDT.DMBC[chrom_idx]
    ploidies = aDT.chrPloidies[chrom_idx]
    DIfilter = aDT.DIByChr[chrom_idx] >= DIthreshold

    RetainedNumer = np.count_nonzero(DIfilter)
    RetainedDenom = DIfilter.size

    # Apply DI filter
    SMf = SM[:, DIfilter]

    # Vectorized state counts per individual
    I4 = np.stack([
        np.count_nonzero(SMf == 0, axis=1),
        np.count_nonzero(SMf == 1, axis=1),
        np.count_nonzero(SMf == 2, axis=1),
        np.count_nonzero(SMf == 3, axis=1),
    ], axis=1)

    # Apply ploidy
    A4 = ploidies[:, None] * I4

    counts0 = A4[:, 0]
    counts1 = A4[:, 1]
    counts2 = A4[:, 2]
    counts3 = A4[:, 3]

    dipDenom = counts1 + counts2 + counts3
    hapDenom = 2 * dipDenom
    stateDenom = dipDenom + counts0

    # Avoid division warnings cleanly
    with np.errstate(divide="ignore", invalid="ignore"):
        HI   = (counts2 + 2 * counts3) / hapDenom
        HOM1 = counts1 / dipDenom
        HET  = counts2 / dipDenom
        HOM2 = counts3 / dipDenom
        U    = counts0 / stateDenom

    summaries = [HI, HOM1, HET, HOM2, U]

    return summaries, RetainedNumer, RetainedDenom

def get_DI_span(aDT):
    minDI=float('inf')
    maxDI=float('-inf')
    for idx, chr  in enumerate(aDT.DMBC):
        minDI=min(minDI,min(aDT.DIByChr[idx]))
        maxDI=max(minDI,max(aDT.DIByChr[idx]))
    return [minDI,maxDI]

from bisect import bisect_left


def fractional_positions_of_multiples(A, delta):
    """
    New, general. ticks solution. SJEB 27/01/2026
    """
    A = np.asarray(A)
    n = len(A)

    values = []
    positions = []

    max_k = A[-1] // delta

    for k in range(1, max_k + 1):
        x = k * delta
        i = bisect_left(A, x)

        # skip values below the first element
        if i == 0:
            continue

        # exact match
        if i < n and A[i] == x:
            pos = float(i)
        else:
            left, right = A[i - 1], A[i]
            pos = (i - 1) + (x - left) / (right - left)

        values.append(x)
        positions.append(pos)
    tick_values = np.array(values)/delta
    tick_positions = np.array(positions)+1
    return np.column_stack((tick_values, tick_positions))
""" new support by SJEB ENDING"""

"""# Polarise and Join
def polarise_n_join(polarisation_data, s_data):
    modified_data = []
    for i in range(len(polarisation_data)):
        polarisation_array = polarisation_data[i]
        s_column = np.array(s_data[i])[:, np.newaxis]
        new_array = np.hstack((polarisation_array, s_column))
        for row in new_array:
            if float(row[0]) ==2:
                row[-1] = StringReplace20(row[-1])
            else:
                row[-1] = row[-1]
        modified_data.append(new_array)
    return modified_data




# MB to ticks:
def chr_mb_ticks(sgl, offset=0, delta=10**6):
    if isinstance(sgl[0], tuple):
        Mb = [x[1] for x in sgl]
    else:
        Mb = sgl
        Mb = Mb.astype(int)
    sites = offset + np.arange(1, len(sgl) + 1)
    Mbticks = np.arange(np.ceil(min(Mb) / delta), np.floor(max(Mb) / delta) + 1)
    Mb_sites_pairs = np.column_stack((Mb, sites))
    Mb_sites_pairs = Mb_sites_pairs[np.lexsort((Mb_sites_pairs[:, 1],))]
    interp_func = interp1d(Mb_sites_pairs[:, 0], Mb_sites_pairs[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    tick_positions = np.round(interp_func(Mbticks * delta)).astype(int)
    tick_values = Mbticks * delta / 10**6
    return np.column_stack((tick_values, tick_positions))


def mb_ticks(gl, delta=10**6):
    chrgl = [list(group) for _, group in pd.groupby(gl, key=lambda x: x[0])]
    lengths = [len(c) for c in chrgl]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ticks = [chr_mb_ticks(chrgl[i], offset=offsets[i], delta=delta) for i in range(len(chrgl))]
    return ticks


def mb1_ticks(gl):
    return mb_ticks(gl, delta=10**6)

def mb2_ticks(gl):
    return mb_ticks(gl, delta=2 * 10**6)


def chr_kb_ticks(sgl, offset=0, delta=10**5):
    if isinstance(sgl[0], tuple):
        Kb = [x[1] for x in sgl]
        Kb = np.array(Kb).astype(float).astype(int)
    else:
        Kb = sgl
        Kb = np.array(Kb).astype(float).astype(int)
    sites = offset + np.arange(1, len(sgl) + 1)
    Kbticks = np.arange(np.ceil(min(Kb) / delta), np.floor(max(Kb) / delta) + 1)
    Kb_sites_pairs = np.column_stack((Kb, sites))
    Kb_sites_pairs = Kb_sites_pairs[np.lexsort((Kb_sites_pairs[:, 1],))]
    interp_func = interp1d(
        Kb_sites_pairs[:, 0], Kb_sites_pairs[:, 1],
        kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    tick_positions = np.round(interp_func(Kbticks * delta)).astype(int)
    tick_values = Kbticks * delta / 10 ** 3  # Convert to kilobases (kb)

    return np.column_stack((tick_values, tick_positions))


def kb_ticks(gl, delta=10 ** 3):
    chrgl = [list(group) for _, group in pd.groupby(gl, key=lambda x: x[0])]
    lengths = [len(c) for c in chrgl]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ticks = [chr_kb_ticks(chrgl[i], offset=offsets[i], delta=delta) for i in range(len(chrgl))]
    return ticks


def kb1_ticks(gl):
    return kb_ticks(gl, delta=10 ** 3)


def kb2_ticks(gl):
    return kb_ticks(gl, delta=2 * 10 ** 3)



# AnnotatedHITally
def AnnotatedHITally(markers):
    string_counts = Counter(markers)
    sorted_counts = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_counts, columns=["Type", "N"])
    total_strings = len(markers)
    df["p"] = df["N"] / total_strings
    df["cum(p)"] = df["p"].cumsum()
    return df
"""

# DiemPlotPrep Class
class DiemPlotPrep:
    def __init__(self, plot_theme, ind_ids, polarised_data, di_threshold, di_column, diemStringPyCol, phys_res, ticks=None, smooth=None):
        self.polarised_data = polarised_data
        self.di_threshold = di_threshold
        self.di_column = di_column
        self.diemStringPyCol = diemStringPyCol
        self.phys_res = phys_res
        self.plot_theme = plot_theme
        self.ind_ids = ind_ids
        self.ticks = ticks
        self.smooth = smooth

        self.diemPlotLabel = None
        self.DIfilteredDATA = None
        self.DIfilteredGenomes = None
        self.DIfilteredHIs = None
        self.DIfilteredBED = None
        self.DIpercent = None
        self.DIfilteredScafRLEs = None
        self.diemDITgenomes = None
        self.DIfilteredGenomes_unsmoothed = None
        self.DIfilteredBED_formatted = None
        self.IndIDs_ordered = None
        self.unit_plot_prep = []
        self.plot_ordered = None
        self.length_of_chromosomes = {}
        self.iris_plot_prep = {}
        self.diemDITgenomes_ordered = None

        self.diem_plot_prep()

    def diem_plot_prep(self):
        """ Perform DI filtering, dithering, and label generation """
        self.filter_data()
        if self.smooth:
            self.kernel_smooth(self.smooth)
        self.diem_dithering()

        self.generate_plot_label(self.plot_theme)

    def format_bed_data(self):
        grouped = {}
        for item in self.DIfilteredBED:
            key, value = item
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)

        self.DIfilteredBED_formatted = [np.array(values) for values in grouped.values()]

        self.plot_ordered = self.DIfilteredHIs
        for a, b in enumerate(self.plot_ordered):
            try:
                self.plot_ordered[a] = (float(b[0]), a + 1)
            except ValueError:
                self.plot_ordered[a] = (np.nan, a + 1)
        self.plot_ordered = sorted(self.plot_ordered, key=lambda x: (np.isnan(x[0]), x[0]))
        # sort the names according to the HIs
        sorted_indices = [index - 1 for _, index in self.plot_ordered]
        # Reorder the names using the sorted indices
        self.IndIDs_ordered = [self.ind_ids[i] for i in sorted_indices]

        start_position = 0
        for bed_data in self.DIfilteredBED_formatted:
            sublist = []
            end_position = start_position + len(bed_data)
            for genome in self.DIfilteredGenomes:
                sublist.append(genome[start_position:end_position])
            sorted_sublist = [sublist[idx] for idx in sorted_indices]
            self.unit_plot_prep.append(sorted_sublist)
            start_position = end_position

        start = 0

        for index, value in enumerate(self.DIfilteredBED_formatted):
            end = start + len(value)  # Current value is the ending point
            length = len(value)  # Calculate the length
            self.length_of_chromosomes[list(grouped.keys())[index]] = [start, end, length]  # Create the dictionary entry
            start = end  # Update the starting point for the next iteration

        for index, bed in enumerate(self.DIfilteredBED_formatted):
            if self.ticks in ['kb', 'KB', 'Kb']:
                x_ticks = chr_kb_ticks(bed).astype(int)
            else:
                x_ticks = chr_mb_ticks(bed).astype(int)
            new_ticks = np.zeros_like(x_ticks)
            starting_point = self.length_of_chromosomes[list(grouped.keys())[index]][0]
            for i, item in enumerate(x_ticks):
                new_value = item[1] + starting_point
                new_ticks[i] = [item[0], new_value]
            self.iris_plot_prep[index + 1] = new_ticks
        self.diemDITgenomes_ordered = [self.diemDITgenomes[i] for i in sorted_indices]

    def filter_data(self):
 #       print('filter_data: di_threshold ',self.di_threshold)
        """ Apply DI threshold filtering on the data """
        if isinstance(self.di_threshold, str):  # No filtering if threshold is a string
            self.DIfilteredDATA = self.polarised_data
        elif isinstance(self.di_threshold, int) or isinstance(self.di_threshold, float):  # Filter above if threshold is just one number
            self.DIfilteredDATA = self.polarised_data[self.polarised_data.DI >= self.di_threshold]
        else:  # Filter within an interval if threshold is a tuple or list
            self.DIfilteredDATA = self.polarised_data[(self.di_threshold[0] <= self.polarised_data.DI) & (self.polarised_data.DI <= self.di_threshold[1])]

        # Extract relevant data after filtering
        self.DIfilteredGenomes = StringTranspose(self.DIfilteredDATA['diem_genotype'])[1:] # slice off the 'S' column
        self.DIfilteredHIs = [pHetErrOnString(genome) for genome in self.DIfilteredGenomes]
        self.DIfilteredBED = self.DIfilteredDATA[['chrom','start']].values.tolist()
        self.DIpercent = round(100 * len(self.DIfilteredDATA) / len(self.polarised_data))
        self.DIfilteredScafRLEs = RichRLE(self.DIfilteredDATA['chrom'].values.tolist())
 #       print("self.DIfilteredScafRLEs[0]: ",self.DIfilteredScafRLEs[0])

    def kernel_smooth(self, scale):
 #       from collections import defaultdict
        scaffold_indices = defaultdict(list)
        for idx, entry in enumerate(self.DIfilteredBED):
            scaffold = entry[0]
            scaffold_indices[scaffold].append(idx)

        split_genomes = []
        for genome in self.DIfilteredGenomes:
            scaffold_dict = {}
            for scaffold, indices in scaffold_indices.items():
                scaffold_str = ''.join([genome[i] for i in indices])
                scaffold_dict[scaffold] = scaffold_str
            split_genomes.append(scaffold_dict)

        scaffold_positions = defaultdict(list)
        for entry in self.DIfilteredBED:
            scaffold = entry[0]  # Scaffold name
            position = entry[1]  # Position
            scaffold_positions[scaffold].append(position)
        scaffold_arrays = {scaffold: np.array(positions) for scaffold, positions in scaffold_positions.items()}

        smoothed_split_genomes = []
        # going through individuals
        for idx, genome in enumerate(split_genomes):
            smoothed_individual_genome = {}
            for key, value in genome.items():
                cleaned_string = value.replace("_", "3")
                integer_list = [int(char) for char in cleaned_string]
                numpy_array_haplo = np.array(integer_list)
#                smooth_output = n_laplace_smooth_one_haplotype(scaffold_arrays[key], numpy_array_haplo, scale)
                smooth_output = laplace_smooth_multiple_haplotypes(scaffold_arrays[key], [numpy_array_haplo], scale)
                string_list = [str(x) for x in smooth_output]
                string_list = ['_' if x == '3' else x for x in string_list]
                result_string = ''.join(string_list)
                smoothed_individual_genome[key] = result_string
            smoothed_split_genomes.append(smoothed_individual_genome)
        self.DIfilteredGenomes_unsmoothed = self.DIfilteredGenomes
        self.DIfilteredGenomes = self._reconstruct_genomes(smoothed_split_genomes, scaffold_indices)

    def _reconstruct_genomes(self, smoothed_split_genomes, scaffold_indices):
        reconstructed_genomes = []

        for individual in smoothed_split_genomes:
            full_genome = ['0'] * len(self.DIfilteredBED)

            for scaffold, indices in scaffold_indices.items():
                scaffold_str = individual[scaffold]
                for i, idx in enumerate(indices):
                    full_genome[idx] = scaffold_str[i]

            reconstructed_genome = ''.join(full_genome)
            reconstructed_genomes.append(reconstructed_genome)

        return reconstructed_genomes


    def diem_dithering(self):
        """ Perform dithering on the filtered data """
        diem_dit_genomes_bed = [list(group) for _, group in groupby(self.DIfilteredBED, key=lambda x: x[0])]
        processed_diemDITgenomes = []
        for chr in diem_dit_genomes_bed:
            length_data = [row[1] for row in chr]
            split_lengths = self.GappedQuotientSplitLengths(length_data, self.phys_res)
            processed_diemDITgenomes.append(split_lengths)
        processed_diemDITgenomes = Flatten(processed_diemDITgenomes)
        diemDITgenomes = []
        for genome in self.DIfilteredGenomes:
            string_take_result = StringTakeList(genome, processed_diemDITgenomes)
            state_count = Map(sStateCount, string_take_result)
            combined = list(zip(state_count, processed_diemDITgenomes))
            # transposed = Transpose([state_count, processed_diemDITgenomes])
            compressed = self.DITcompress(combined)
            lengths = self.Lengths2StartEnds(compressed)
            diemDITgenomes.append(lengths)

        self.diemDITgenomes = diemDITgenomes


    def generate_plot_label(self, plot_theme):
        """ Generate the label for the plot """
        self.diemPlotLabel = f"{plot_theme} @ DI = {self.di_threshold}: {len(self.DIfilteredGenomes)} sites ({self.DIpercent}%)."

    @staticmethod
    def GappedQuotientSplit(lst, Q):
        """
        Splits the list `lst` into sublists where consecutive elements share the same quotient when divided by `Q`.
        """
        quotients = [x // Q for x in lst]

        groups = []
        current_group = [lst[0]]

        for i in range(1, len(lst)):
            if quotients[i] == quotients[i - 1]:
                current_group.append(lst[i])
            else:
                groups.append(current_group)
                current_group = [lst[i]]

        groups.append(current_group)
        return groups

    def GappedQuotientSplitLengths(self, lst, Q):
        """
        Returns the lengths of the sublists produced by `gapped_quotient_split`.
        """
        return Map(len, self.GappedQuotientSplit(lst, Q))

    @staticmethod
    def normalize_4list(lst):
        """
        Normalizes a 4list by converting each element to its ratio of the total sum.
        Uses Fraction for precise comparison without floating-point errors.
        """
        total = sum(lst)
        if total == 0:
            return tuple(0 for _ in lst)  # Handle case where total is 0
        return tuple(Fraction(x, total) for x in lst)

    def DITcompress(self, DITl):
        """
        Compresses the list of {4list, length} tuples.
        """
        grouped_data = [list(group) for _, group in groupby(DITl, key=lambda x: self.normalize_4list(x[0]))]
        final_data = []
        for group in grouped_data:
            summed_states = [sum(x) for x in zip(*(item[0] for item in group))]
            summed_value = sum(item[1] for item in group)
            result = (summed_states, summed_value)
            final_data.append(result)
        return final_data

    @staticmethod
    def Lengths2StartEnds(stateNlen):
        lengths = [x[1] for x in stateNlen]
        ends = np.cumsum(lengths)

        # Calculate the start positions (end positions minus length plus 1)
        starts = ends - np.array(lengths) + 1

        # Combine states, starts, and ends into a list of triplets
        result = [(state, int(start), int(end)) for (state, start, end) in zip([x[0] for x in stateNlen], starts, ends)]

        return result





# DiemRectangleDiagram
diemColours = [
    'white',
    mcolors.to_hex((128/255, 0, 128/255)),  # RGBColor[128/255, 0, 128/255] - Purple
    mcolors.to_hex((255/255, 229/255, 0)),  # RGBColor[255/255, 229/255, 0] - Yellow
    mcolors.to_hex((0, 128/255, 128/255))   # RGBColor[0, 128/255, 128/255] - Teal
]
char_to_index = {
    '_': 0,
    'U': 0,
    '0': 1,
    '1': 2,
    '2': 3
}





# Pairwise distance graph:
class PWC:
    def __init__(self, PWCtallyer, PWCweight, input_path, output_path_results, output_path_heatmap, labels):
        self.PWCtallyer = PWCtallyer
        self.PWCweight = PWCweight
        self.input_path = input_path
        self.output_path_results = output_path_results
        self.output_path_heatmap = output_path_heatmap
        self.U012 = "_012"
        self.labels = labels
        self.PWCtallyer = []
        for i in range(len(self.U012)):
            for j in range(i, len(self.U012)):
                PWCtallyer.append(self.ASJ([self.U012[i], self.U012[j]]))
                if i != j:
                    PWCtallyer.append(self.ASJ([self.U012[j], self.U012[i]]))
        self.PWCweight = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 1, 0]

    @staticmethod
    def SJ(l):
        return "".join(map(str, l))

    @staticmethod
    def ASJ(s):
        return PWC.SJ(s)


    def PWCtally(self, l):
        return [Counter(self.PWCtallyer + l)[k] - 1 for k in self.PWCtallyer]

    def diemDistancePerSite(self, g1, g2):
        pwct = self.PWCtally(["".join(pair) for pair in zip(g1, g2)])
        return np.sum(np.array(pwct) * np.array(self.PWCweight)) / np.sum(pwct[7:])

    @staticmethod
    def CombineRules(ss):
        rules = {
            "00": "0",
            "22": "2",
            "02": "1",
            "20": "1"
        }
        return rules.get(ss, "_")

    @staticmethod
    def diemCombineSites(g1, g2):
        return PWC.ASJ([PWC.CombineRules("".join(pair)) for pair in zip(g1, g2)])

    @staticmethod
    def diemOffspringDistance(p1, p2, o):
        return PWC.diemDistancePerSite(PWC.diemCombineSites(p1, p2), o)

    def pwc_graph(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            RawSyphGenomes = f.read()
        RawSyphGenomes = [s.replace('\n', '') for s in RawSyphGenomes.split("S") if s]
        PWCheatMap = np.zeros((len(RawSyphGenomes), len(RawSyphGenomes)))

        # Calculate pairwise distances
        for i in range(len(RawSyphGenomes)):
            for j in range(len(RawSyphGenomes)):
                PWCheatMap[i, j] = self.diemDistancePerSite(RawSyphGenomes[i], RawSyphGenomes[j])
        np.savetxt(self.output_path_results, PWCheatMap, delimiter=",")
        custom_cmap = LinearSegmentedColormap.from_list("soft_coolwarm", ["#1e90ff", 'white', "#fff266", "#ff1a1a"])
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(PWCheatMap, cmap=custom_cmap, xticklabels=self.labels, yticklabels=self.labels,
                 cbar_kws={"shrink": 1, "fraction": 0.045, "pad": 0.04})
        ax.set_aspect('equal', adjustable='box')
        plt.xticks(rotation=-90)
        plt.savefig(self.output_path_heatmap, format="png", dpi=500)


#________________________________________ START GenomeSummariesPlot ___________________
from matplotlib.widgets import Button, Slider

class GenomeSummaryPlot:
    def __init__(self, dPol):
        self.dPol = dPol

        # ---- initial state ----
        self.indNameFont = 6
        self.indHIorder = np.arange(len(dPol.indNames))

        # initial summaries
        self.summaries, self.DInumer, self.DIdenom = genomes_summary_given_DI(
            dPol, float("-inf")
        )
        self.prop = self.DInumer / self.DIdenom

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(11, 4))

        colours = Flatten(
            [['red'], diemColours[1:], ['gray']]
        )

        self.lines = []
        for summary, colour in zip(self.summaries, colours):
            line, = self.ax.plot(summary, color=colour, marker='.')
            self.lines.append(line)

        self.ax.legend(['HI', 'HOM1', 'HET', 'HOM2', 'U'])
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Genomes summaries; no DI filter')
        self.ax.tick_params(axis='x', rotation=55)

        self._update_xticks()

        # ---- widgets ----
        self._init_widgets()
        
        # ---- coordinate display ----
        self._install_format_coord()

    # ---------------- helpers ----------------

    def _update_xticks(self):
        self.ax.set_xticks(
            np.arange(len(self.dPol.indNames)),
            np.array(self.dPol.indNames)[self.indHIorder],
            rotation=55,
            fontsize=self.indNameFont,
            horizontalalignment='right'
        )
    def _install_format_coord(self):
        n = len(self.dPol.indNames)
        tolerance = 0.03  # vertical proximity in y-units
    
        def format_coord(x, y):
            # Quiet fallback (keeps ipympl stable)
            fallback = "\u2007" * 30
    
            # Nearest x index
            i = int(round(x))
            if i < 0 or i >= n:
                return fallback
    
            # Check all summary lines for proximity
            for summary in self.summaries:
                y0 = summary[self.indHIorder][i]
                if abs(y - y0) < tolerance:
                    return f"IndID: {self.dPol.indNames[self.indHIorder[i]]}"
    
            return fallback
    
        self.ax.format_coord = format_coord
    
    # ---------------- widgets ----------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        self.fig.subplots_adjust(bottom=0.3)

        # DI slider
        DI_box = self.fig.add_axes([0.2, 0.1, 0.65, 0.03])
        self.DI_slider = Slider(
            ax=DI_box,
            label='DI',
            valmin=DI_span[0],
            valmax=DI_span[1],
            valinit=DI_span[0],
        )
        self.DI_slider.on_changed(self.DIupdate)

        # Font slider
        FONT_box = self.fig.add_axes([0.25, 0.025, 0.1, 0.04])
        self.FONT_slider = Slider(
            ax=FONT_box,
            label='xLabels font',
            valmin=1,
            valmax=16,
            valinit=self.indNameFont,
        )
        self.FONT_slider.on_changed(self.FONTupdate)

        # Reorder button
        reorderBox = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reo_button = Button(
            reorderBox,
            'Reorder by HI',
            hovercolor='0.975',
            color='red'
        )
        self.reo_button.on_clicked(self.reorder)

    # ---------------- callbacks ----------------

    def DIupdate(self, val):
        self.summaries, self.DInumer, self.DIdenom = genomes_summary_given_DI(
            self.dPol, val
        )
        self.prop = self.DInumer / self.DIdenom

        self.ax.set_title(
            "Genomes summaries DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(val, self.DInumer, 100 * self.prop)
        )

        for line, summary in zip(self.lines, self.summaries):
            line.set_ydata(summary[self.indHIorder])

        self.fig.canvas.draw_idle()

    def reorder(self, event):
        self.indHIorder = np.argsort(self.summaries[0])  # HI

        self._update_xticks()

        for line, summary in zip(self.lines, self.summaries):
            line.set_ydata(summary[self.indHIorder])

        self.fig.canvas.draw_idle()

    def FONTupdate(self, val):
        self.indNameFont = val
        self._update_xticks()
        self.fig.canvas.draw_idle()
#________________________________________ END GenomeSummariesPlot ___________________



#________________________________________ START GenomeMultiSummaryPlot ___________________

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class GenomeMultiSummaryPlot:
    def __init__(self, dPol, chrom_indices, max_cols=3):
        self.dPol = dPol
        self.max_cols = max_cols

        # ---- validate chromosomes ----
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)

        # ---- ordering state ----
        self.indNameFont = 6

        # whole-genome HI (authoritative ordering reference)
        global_summaries, _, _ = genomes_summary_given_DI(
            self.dPol, float("-inf")
        )
        self.global_HI = global_summaries[0]
        self.indHIorder = np.argsort(self.global_HI)

        # ---- initial summaries ----
        self._compute_summaries(float("-inf"))

        # ---- grid layout ----
        n_plots = len(self.chrom_indices)
        n_cols = min(self.max_cols, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

        fig_w = 4.5 * n_cols
        fig_h = 3.5 * n_rows

        self.fig, self.axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharey=True
        )

        self.fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.92,
            bottom=0.45,   # <-- important: room for sliders.
            hspace=0.35,
            wspace=0.25
        )

        # ---- draw plots ----
        self.lines = {}
        axes_flat = self.axes.flatten()

        colours = Flatten(
            [['red'], diemColours[1:], ['gray']]
        )

        for ax, chrom_idx in zip(axes_flat, self.chrom_indices):
            summaries = self.chrom_summaries[chrom_idx]

            chrom_lines = []
            for summary, colour in zip(summaries, colours):
                line, = ax.plot(
                    summary[self.indHIorder],
                    color=colour,
                    marker='.',
                    linewidth=0.8
                )
                chrom_lines.append(line)

            self.lines[chrom_idx] = chrom_lines

            ax.set_ylim(0, 1)
            ax.set_title(f"Chr {chrom_idx}", fontsize=10)
            ax.tick_params(axis='x', rotation=55)

            ax.set_xticks(
                np.arange(len(self.dPol.indNames)),
                np.array(self.dPol.indNames)[self.indHIorder],
                fontsize=self.indNameFont,
                ha='right'
            )

        # hide unused axes
        for ax in axes_flat[len(self.chrom_indices):]:
            ax.axis("off")

        # legend once
        axes_flat[0].legend(
            ['HI', 'HOM1', 'HET', 'HOM2', 'U'],
            fontsize=8,
            frameon=False
        )

        # ---- widgets ----
        self._init_widgets()

        plt.show()

    # ==================================================
    # Validation
    # ==================================================

    def _validate_chrom_indices(self, chrom_indices):
        max_idx = len(self.dPol.DMBC) - 1
        valid, rejected = [], []

        for idx in chrom_indices:
            if isinstance(idx, (int, np.integer)) and 0 <= idx <= max_idx:
                valid.append(int(idx))
            else:
                rejected.append(idx)

        if rejected:
            print(
                "GenomeMultiSummaryPlot: rejected chromosome indices:",
                rejected
            )

        if not valid:
            raise ValueError(
                "GenomeMultiSummaryPlot: no valid chromosome indices."
            )

        return valid

    # ==================================================
    # Computation
    # ==================================================

    def _compute_summaries(self, DIval):
        self.chrom_summaries = {}
        self.DInumer = {}
        self.DIdenom = {}

        for idx in self.chrom_indices:
            summaries, DIn, DId = genomes_summary_given_DI_by_chromosome(
                self.dPol, DIval, idx
            )
            self.chrom_summaries[idx] = summaries
            self.DInumer[idx] = DIn
            self.DIdenom[idx] = DId

    # ==================================================
    # Widgets
    # ==================================================

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        # DI slider
        ax_DI = self.fig.add_axes([0.15, 0.18, 0.7, 0.03])
        self.DI_slider = Slider(
            ax_DI, "DI",
            DI_span[0], DI_span[1],
            valinit=DI_span[0]
        )
        self.DI_slider.on_changed(self._on_DI_change)

        # font slider
        ax_FS = self.fig.add_axes([0.20, 0.12, 0.35, 0.03])
        self.FONT_slider = Slider(
            ax_FS,
            "Label font",
            4, 16,
            valinit=self.indNameFont,
            valstep=1
        )
        self.FONT_slider.on_changed(self._on_font_change)

        # reorder button
        ax_RE = self.fig.add_axes([0.62, 0.115, 0.30, 0.045])
        self.reorder_button = Button(
            ax_RE,
            "Reorder by global HI",
            hovercolor="0.95",
            color="red"
        )
        self.reorder_button.on_clicked(self._on_reorder)

    # ==================================================
    # Callbacks
    # ==================================================

    def _on_DI_change(self, val):
        self._compute_summaries(val)

        for idx in self.chrom_indices:
            summaries = self.chrom_summaries[idx]
            for line, summary in zip(self.lines[idx], summaries):
                line.set_ydata(summary[self.indHIorder])

        self.fig.canvas.draw_idle()

    def _on_font_change(self, val):
        self.indNameFont = int(val)
        labels = np.array(self.dPol.indNames)[self.indHIorder]

        for ax in self.axes.flatten()[:len(self.chrom_indices)]:
            ax.set_xticklabels(labels, fontsize=self.indNameFont)

        self.fig.canvas.draw_idle()

    def _on_reorder(self, event=None):
        """
        Reorder individuals by whole-genome HI
        at the CURRENT DI threshold.
        """
    
        summaries, _, _ = genomes_summary_given_DI(
            self.dPol, self.DI_slider.val
        )
    
        global_HI = summaries[0]
        self.indHIorder = np.argsort(global_HI)
    
        labels = np.array(self.dPol.indNames)[self.indHIorder]
    
        for idx in self.chrom_indices:
            summaries = self.chrom_summaries[idx]
            for line, summary in zip(self.lines[idx], summaries):
                line.set_ydata(summary[self.indHIorder])
    
        for ax in self.axes.flatten()[:len(self.chrom_indices)]:
            ax.set_xticks(
                np.arange(len(labels)),
                labels,
                fontsize=self.indNameFont,
                ha="right"
            )
    
        self.fig.canvas.draw_idle()

#________________________________________ END GenomeMultiSummaryPlot ___________________


#________________________________________ START GenomicDeFinettiPlot ___________________

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgb

class GenomicDeFinetti:
    def __init__(self, dPol):
        self.dPol = dPol

        # ---- initial state ----
        self.marker_size = 60
        self.indHIorder = np.arange(len(dPol.indNames))

        # initial summaries
        self.summaries, self.DInumer, self.DIdenom = genomes_summary_given_DI(
            dPol, float("-inf")
        )

        # unpack summaries
        # order: HI, HOM1, HET, HOM2, U
        self.HOM1 = self.summaries[1]
        self.HET  = self.summaries[2]
        self.HOM2 = self.summaries[3]
        self.U    = self.summaries[4]

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self._setup_axes()
#        self._update_title(float("-inf"))
        self.ax.set_title('Genomic de Finetti; no DI filter')

        # background: triangle + HW curve
        self._draw_triangle()
        self._draw_hwe_curve()

        # points
        self.scatter = self._draw_points()

        # widgets
        self._init_widgets()

        # coordinate display
        self._install_format_coord()

        plt.show()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _to_triangle_coords(hom1, het, hom2):
        x = hom2 + 0.5 * het
        y = (np.sqrt(3) / 2) * het
        return x, y

    def _update_title(self, DIval):
        prop = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0

        self.ax.set_title(
            "Genomic de Finetti plot  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(DIval, self.DInumer, 100 * prop),
            fontsize=12,
            pad=12
        )
    
    # --------------------------------------------------
    # Axes / background
    # --------------------------------------------------

    def _setup_axes(self):
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax.set_title("Genomic de Finetti plot")

    def _draw_triangle(self):
        h = np.sqrt(3) / 2
        triangle = np.array([[0, 0], [1, 0], [0.5, h]])
        self.ax.add_patch(
            Polygon(triangle, closed=True, fill=False, lw=1.2, color="black")
        )

        self.ax.text(0, -0.04, "HOM1", ha="center", va="top", fontsize=9)
        self.ax.text(1, -0.04, "HOM2", ha="center", va="top", fontsize=9)
        self.ax.text(0.5, h + 0.03, "HET", ha="center", va="bottom", fontsize=9)

    def _draw_hwe_curve(self):
        # Hardy–Weinberg: p^2, 2pq, q^2
        p = np.linspace(0, 1, 400)
        hom1 = p**2
        het  = 2 * p * (1 - p)
        hom2 = (1 - p)**2

        x, y = self._to_triangle_coords(hom1, het, hom2)
        self.ax.plot(x, y, color="black", lw=0.8, alpha=0.5)

    # --------------------------------------------------
    # Points
    # --------------------------------------------------

    def _blend_colours(self):
        """
        Blend colours using HOM1, HET, HOM2, U.
        Notice: HI is deliberately excluded.
        """
    
        # shape (n_individuals, 4)
        weights = np.column_stack([
            self.HOM1,
            self.HET,
            self.HOM2,
            self.U
        ])
    
        # exactly 4 colours, in the same order
        base_colours = np.array([
            to_rgb(diemColours[1]),  # HOM1
            to_rgb(diemColours[2]),  # HET
            to_rgb(diemColours[3]),  # HOM2
            to_rgb(diemColours[0]),  # U
        ])  # shape (4, 3)
    
        rgb = weights @ base_colours
        rgb = np.clip(rgb, 0.0, 1.0)
    
        return rgb

    
    def _draw_points(self):
        x, y = self._to_triangle_coords(
            self.HOM1[self.indHIorder],
            self.HET[self.indHIorder],
            self.HOM2[self.indHIorder],
        )

        colours = self._blend_colours()[self.indHIorder]

        return self.ax.scatter(
            x, y,
            s=self.marker_size,
            c=colours,
            edgecolor="black",
            linewidth=0.3,
        )

    def _update_points(self):
        self.HOM1 = self.summaries[1]
        self.HET  = self.summaries[2]
        self.HOM2 = self.summaries[3]
        self.U = self.summaries[4]

        x, y = self._to_triangle_coords(
            self.HOM1[self.indHIorder],
            self.HET[self.indHIorder],
            self.HOM2[self.indHIorder],
        )

        self.scatter.set_offsets(np.column_stack([x, y]))
        self.scatter.set_facecolors(self._blend_colours()[self.indHIorder])

    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)
        self.fig.subplots_adjust(bottom=0.25)

        # DI slider
        ax_DI = self.fig.add_axes([0.15, 0.12, 0.7, 0.03])
        self.DI_slider = Slider(
            ax_DI, "DI",
            DI_span[0], DI_span[1],
            valinit=DI_span[0]
        )
        self.DI_slider.on_changed(self.DIupdate)

        # size slider
        ax_SZ = self.fig.add_axes([0.25, 0.06, 0.5, 0.03])
        self.size_slider = Slider(
            ax_SZ, "Symbol size",
            10, 300,
            valinit=self.marker_size
        )
        self.size_slider.on_changed(self.SIZEupdate)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def DIupdate(self, val):
        self.summaries, self.DInumer, self.DIdenom = genomes_summary_given_DI(
            self.dPol, val
        )
    
        # unpack again (important!)
        HI, HOM1, HET, HOM2, U = self.summaries
        self.HOM1 = HOM1
        self.HET  = HET
        self.HOM2 = HOM2
        self.U    = U
    
        self._update_points()
        self._update_title(val)
    
        self.fig.canvas.draw_idle()

    
    def SIZEupdate(self, val):
        self.marker_size = val
        self.scatter.set_sizes(np.full(len(self.dPol.indNames), val))
        self.fig.canvas.draw_idle()

    # --------------------------------------------------
    # Coordinate display (robust, no events)
    # --------------------------------------------------

    def _install_format_coord(self):
        n = len(self.dPol.indNames)
        tol = 0.03  # proximity in triangle units

        def format_coord(x, y):
            fallback = "\u2007" * 30

            pts = self.scatter.get_offsets()
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            i = np.argmin(d)

            if d[i] < tol:
                return f"IndID: {self.dPol.indNames[self.indHIorder[i]]}"

            return fallback

        self.ax.format_coord = format_coord
#________________________________________ END GenomicDeFinetti ___________________


#________________________________________ START GenomicMultiDeFinetti ___________________

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgb


class GenomicMultiDeFinetti:
    def __init__(self, dPol, chrom_indices, max_cols=3):
        self.dPol = dPol
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)
        self.max_cols = max_cols

        # ---- state ----
        self.marker_size = 60
        self.indHIorder = np.arange(len(dPol.indNames))

        # ---- initial summaries ----
        self._compute_summaries(float("-inf"))

        # ---- grid layout ----
        self.n_plots = len(self.chrom_indices)
        self.n_cols = min(self.max_cols, self.n_plots)
        self.n_rows = math.ceil(self.n_plots / self.n_cols)

        fig_w = 3.8 * self.n_cols
        fig_h = 4.0 * self.n_rows

        self.fig, self.axes = plt.subplots(
            self.n_rows,
            self.n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False
        )

        self.fig.subplots_adjust(
            left=0.05,
            right=0.98,
            top=0.93,
            bottom=0.25,
            wspace=0.25,
            hspace=0.35
        )

        # ---- draw all subplots ----
        self.scatters = {}
        axes_flat = self.axes.flatten()

        for ax, chrom_idx in zip(axes_flat, self.chrom_indices):
            self._setup_axes(ax)
            self._draw_triangle(ax)
            self._draw_hwe_curve(ax)

            scatter = self._draw_points(ax, chrom_idx)
            self.scatters[chrom_idx] = scatter

            self._install_format_coord() # for hover

            ax.set_title(
#                f"Chr {chrom_idx}",
                self.dPol.chrNames[chrom_idx],
                fontsize=10,
                pad=6
            )

        # hide unused axes
        for ax in axes_flat[self.n_plots:]:
            ax.axis("off")

        # ---- widgets ----
        self._init_widgets()

        plt.show()

    # ==================================================
    # Helpers
    # ==================================================

    def _validate_chrom_indices(self, chrom_indices):
        max_idx = len(self.dPol.DMBC) - 1
    
        valid = []
        rejected = []
    
        for idx in chrom_indices:
            if isinstance(idx, (int, np.integer)) and 0 <= idx <= max_idx:
                valid.append(int(idx))
            else:
                rejected.append(idx)
    
        if rejected:
            print(
                "GenomicMultiDeFinetti: rejected chromosome indices:",
                rejected
            )
    
        if not valid:
            raise ValueError(
                "GenomicMultiDeFinetti: no valid chromosome indices to plot."
            )
    
        return valid

    @staticmethod
    def _to_triangle_coords(hom1, het, hom2):
        x = hom2 + 0.5 * het
        y = (np.sqrt(3) / 2) * het
        return x, y

    # ==================================================
    # Axes / background
    # ==================================================

    def _setup_axes(self, ax):
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_triangle(self, ax):
        h = np.sqrt(3) / 2
        tri = np.array([[0, 0], [1, 0], [0.5, h]])
        ax.add_patch(
            Polygon(tri, closed=True, fill=False, lw=1.2, color="black")
        )

        ax.text(0, -0.04, "HOM1", ha="center", va="top", fontsize=8)
        ax.text(1, -0.04, "HOM2", ha="center", va="top", fontsize=8)
        ax.text(0.5, h + 0.03, "HET", ha="center", va="bottom", fontsize=8)

    def _draw_hwe_curve(self, ax):
        p = np.linspace(0, 1, 400)
        hom1 = p**2
        het = 2 * p * (1 - p)
        hom2 = (1 - p)**2

        x, y = self._to_triangle_coords(hom1, het, hom2)
        ax.plot(x, y, color="black", lw=0.8, alpha=0.5)

    # ==================================================
    # Summaries (PER INDIVIDUAL, PER CHROMOSOME)
    # ==================================================

    def _compute_summaries(self, DIval):
        """
        Per-individual HOM1 / HET / HOM2 / U
        computed separately for each chromosome.
        """
        self.chrom_summaries = {}

        for idx in self.chrom_indices:
            chr_DM = self.dPol.DMBC[idx]
            DIvals = self.dPol.DIByChr[idx]
            mask = DIvals >= DIval

            n_ind = chr_DM.shape[0]

            HOM1 = np.zeros(n_ind)
            HET = np.zeros(n_ind)
            HOM2 = np.zeros(n_ind)
            U = np.zeros(n_ind)

            for i, indiv in enumerate(chr_DM):
                vals = indiv[mask]
                total = len(vals)

                if total > 0:
                    HOM1[i] = np.count_nonzero(vals == 1) / total
                    HET[i]  = np.count_nonzero(vals == 2) / total
                    HOM2[i] = np.count_nonzero(vals == 3) / total
                    U[i]    = np.count_nonzero(vals == 0) / total

            self.chrom_summaries[idx] = (HOM1, HET, HOM2, U)

    # ==================================================
    # Colours and points
    # ==================================================

    def _blend_colours(self, HOM1, HET, HOM2, U):
        weights = np.column_stack([HOM1, HET, HOM2, U])

        base_colours = np.array([
            to_rgb(diemColours[1]),  # HOM1
            to_rgb(diemColours[2]),  # HET
            to_rgb(diemColours[3]),  # HOM2
            to_rgb(diemColours[0]),  # U
        ])

        rgb = weights @ base_colours
        return np.clip(rgb, 0.0, 1.0)

    def _draw_points(self, ax, chrom_idx):
        HOM1, HET, HOM2, U = self.chrom_summaries[chrom_idx]

        x, y = self._to_triangle_coords(
            HOM1[self.indHIorder],
            HET[self.indHIorder],
            HOM2[self.indHIorder],
        )

        colours = self._blend_colours(
            HOM1[self.indHIorder],
            HET[self.indHIorder],
            HOM2[self.indHIorder],
            U[self.indHIorder],
        )

        return ax.scatter(
            x, y,
            s=self.marker_size,
            c=colours,
            edgecolor="black",
            linewidth=0.3,
        )

    def _update_points(self):
        for idx, scatter in self.scatters.items():
            HOM1, HET, HOM2, U = self.chrom_summaries[idx]

            x, y = self._to_triangle_coords(
                HOM1[self.indHIorder],
                HET[self.indHIorder],
                HOM2[self.indHIorder],
            )

            scatter.set_offsets(np.column_stack([x, y]))
            scatter.set_facecolors(
                self._blend_colours(
                    HOM1[self.indHIorder],
                    HET[self.indHIorder],
                    HOM2[self.indHIorder],
                    U[self.indHIorder],
                )
            )

    # ==================================================
    # Hover
    # ==================================================
    def _install_format_coord(self):
        tol = 0.03  # proximity in triangle units
    
        for chrom_idx, scatter in self.scatters.items():
            ax = scatter.axes
    
            def make_formatter(ax, scatter, chrom_idx):
                def format_coord(x, y):
                    pts = scatter.get_offsets()
                    if len(pts) == 0:
                        return ""
    
                    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
                    i = np.argmin(d)
    
                    if d[i] < tol:
                        ind = self.indHIorder[i]
                        return (
                            f"Chr {chrom_idx}   "
                            f"IndID: {self.dPol.indNames[ind]}"
                        )
    
                    return ""
    
                return format_coord
    
            ax.format_coord = make_formatter(ax, scatter, chrom_idx)
    # ==================================================
    # Widgets
    # ==================================================

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        # DI slider
        ax_DI = self.fig.add_axes([0.15, 0.14, 0.7, 0.03])
        self.DI_slider = Slider(
            ax_DI, "DI",
            DI_span[0], DI_span[1],
            valinit=DI_span[0]
        )
        self.DI_slider.on_changed(self._on_DI_change)

        # size slider
        ax_SZ = self.fig.add_axes([0.25, 0.08, 0.5, 0.03])
        self.size_slider = Slider(
            ax_SZ, "Symbol size",
            10, 300,
            valinit=self.marker_size
        )
        self.size_slider.on_changed(self._on_size_change)

    # ==================================================
    # Callbacks
    # ==================================================

    def _on_DI_change(self, val):
        self._compute_summaries(val)
        self._update_points()
        self.fig.canvas.draw_idle()

    def _on_size_change(self, val):
        self.marker_size = val
        for scatter in self.scatters.values():
            scatter.set_sizes(np.full(len(self.dPol.indNames), val))
        self.fig.canvas.draw_idle()
#________________________________________ END GenomicMultiDeFinetti ___________________


#________________________________________ START GenomicContributions ___________________

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider

class GenomicContributions:
    def __init__(self, dPol):
        self.dPol = dPol
        self.fontsize = 8

        # initial computation
        self._compute_contributions(float("-inf"))

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.format_coord = None
        self.fig.subplots_adjust(bottom=0.40, right=0.85)
        self.ax.set_title('Genomic Contributions; no DI filter')

        self._draw_bars()
        self._init_widgets()

        plt.show()

    # --------------------------------------------------
    # Core computation
    # --------------------------------------------------

    def _compute_contributions(self, DIval):
        """
        Per-chromosome proportions of:
        HOM1, HET, HOM2, U, excluded (DI-filtered)
        """

        n_chr = len(self.dPol.DMBC)

        self.chrom_labels = []
        self.props = np.zeros((n_chr, 5))  # HOM1, HET, HOM2, U, excluded

        for i, chr_DM in enumerate(self.dPol.DMBC):
            chr_name = self.dPol.chrNames[i]
            chr_name = chr_name.replace("chromosome_", "Chr ") # Nickname Chromosomes
            DIvals = self.dPol.DIByChr[i]

            mask = DIvals >= DIval

            total_sites = len(DIvals)
            kept_sites = np.count_nonzero(mask)
            excluded_sites = total_sites - kept_sites

            # A4-style accumulation across individuals
            A4 = np.zeros(4, dtype=int)  # [U, HOM1, HET, HOM2]

            for indiv in chr_DM:
                vals = indiv[mask]
                A4[0] += np.count_nonzero(vals == 0)
                A4[1] += np.count_nonzero(vals == 1)
                A4[2] += np.count_nonzero(vals == 2)
                A4[3] += np.count_nonzero(vals == 3)

            U, HOM1, HET, HOM2 = A4

            n_indiv = chr_DM.shape[0]
            total_genotypes = total_sites * n_indiv
            
            self.props[i, :] = [
                HOM1 / total_genotypes,
                HET  / total_genotypes,
                HOM2 / total_genotypes,
                U    / total_genotypes,
                excluded_sites / total_sites,
            ]
            
            self.chrom_labels.append(chr_name)

        self.current_DI = DIval

    # --------------------------------------------------
    # Drawing
    # --------------------------------------------------

    def _draw_bars(self):
        self.ax.clear()

        x = np.arange(len(self.chrom_labels))
        bottoms = np.zeros(len(x))

        colours = [
            diemColours[1],  # HOM1
            diemColours[2],  # HET
            diemColours[3],  # HOM2
            "lightgray",                # U
            "white",                    # excluded
        ]

        labels = ["HOM1", "HET", "HOM2", "U", "<DI"]

        for i in range(5):
            self.ax.bar(
                x,
                self.props[:, i],
                bottom=bottoms,
                color=colours[i],
                edgecolor="black" if i == 4 else None,
                linewidth=0.4 if i == 4 else 0,
                label=labels[i],
            )
            bottoms += self.props[:, i]

        self.ax.set_xlim(-0.5, len(x) - 0.5)
        self.ax.set_ylim(0, 1)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(
            self.chrom_labels,
            rotation=90,
            fontsize=self.fontsize,
            ha="center",
        )

        self.ax.set_ylabel("Proportion of SNVs")

        self.ax.set_title(
            "Genomic contributions by chromosome  DI ≥ {:.2f}".format(self.current_DI),
            pad=12
        )

        # legend outside plot
        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            frameon=False,
        )

        self.fig.canvas.draw_idle()

    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        # DI slider (lower, wide)
        ax_DI = self.fig.add_axes([0.15, 0.20, 0.70, 0.03])
        self.DI_slider = Slider(
            ax_DI,
            "DI",
            DI_span[0],
            DI_span[1],
            valinit=DI_span[0],
        )
        self.DI_slider.on_changed(self.DIupdate)

        # Font size slider (narrow, left)
        ax_FS = self.fig.add_axes([0.15, 0.13, 0.35, 0.03])
        self.font_slider = Slider(
            ax_FS,
            "Label font size",
            4,
            16,
            valinit=self.fontsize,
            valstep=1,
        )
        self.font_slider.on_changed(self.FONTupdate)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def DIupdate(self, val):
        self._compute_contributions(val)
        self._draw_bars()

    def FONTupdate(self, val):
        self.fontsize = int(val)
        self._draw_bars()
#________________________________________ END GenomicContributions__________________

#________________________________________ START DiemPlotPrep__________________
#from collections import defaultdict

#from itertools import groupby
# DiemPlotPrep Class
class DiemPlotPrep:
    def __init__(self, plot_theme, ind_ids, polarised_data, di_threshold, di_column, diemStringPyCol, genome_pixels, ticks=None, smooth=None):
        self.polarised_data = polarised_data
        self.di_threshold = di_threshold
        self.di_column = di_column
        self.diemStringPyCol = diemStringPyCol
        self.genome_pixels = genome_pixels
        self.plot_theme = plot_theme
        self.ind_ids = ind_ids
        self.ticks = ticks
        self.smooth = smooth

        self.diemPlotLabel = None
        self.DIfilteredDATA = None
        self.DIfilteredGenomes = None
        self.DIfilteredHIs = None
        self.DIfilteredBED = None
        self.DIpercent = None
        self.DIfilteredScafRLEs = None
        self.diemDITgenomes = None
        self.DIfilteredGenomes_unsmoothed = None
        self.DIfilteredBED_formatted = None
        self.IndIDs_ordered = None
        self.unit_plot_prep = []
        self.plot_ordered = None
        self.length_of_chromosomes = {}
        self.iris_plot_prep = {}
        self.diemDITgenomes_ordered = None
        self.nBasesDithered = None

        self.diem_plot_prep()

    def diem_plot_prep(self):
        """ Perform DI filtering, dithering, and label generation """
        self.filter_data()
        if self.smooth:
            self.kernel_smooth(self.smooth)
        self.diem_dithering()

        self.generate_plot_label(self.plot_theme)
        self.format_bed_data() 

    # format_bed_data reworked by ChatGPT 5.2 for speed, but mostly clarity
    def format_bed_data(self):
        # -------------------------------------------------
        # 1. Compute ordering by HI (vectorised + clearer)
        # -------------------------------------------------
        HI_values = np.array([
            float(b[0]) if b[0] is not None else np.nan
            for b in self.DIfilteredHIs
        ])
    
        # stable sort: NaNs last
        sorted_indices = np.argsort(
            np.isnan(HI_values), kind="stable"
        )
        sorted_indices = sorted_indices[np.argsort(HI_values[sorted_indices], kind="stable")]
    
        self.plot_ordered = list(zip(HI_values[sorted_indices], sorted_indices + 1))
        self.IndIDs_ordered = [self.ind_ids[i] for i in sorted_indices]
        self.diemDITgenomes_ordered = [self.diemDITgenomes[i] for i in sorted_indices]
    
        # -------------------------------------------------
        # 2. Prepare unit_plot_prep (slice once, reuse)
        # -------------------------------------------------
        self.unit_plot_prep = []
        
        start = 0
        for bed_data in self.DIfilteredBED_formatted:
            end = start + len(bed_data)
        
            sublist = [genome[start:end] for genome in self.DIfilteredGenomes]
            self.unit_plot_prep.append([sublist[idx] for idx in sorted_indices])
        
            start = end
    
    
    def filter_data(self):
        """ Apply DI threshold filtering on the data """
        if isinstance(self.di_threshold, str):  # No filtering if threshold is a string
            self.DIfilteredDATA = self.polarised_data
        elif isinstance(self.di_threshold, int) or isinstance(self.di_threshold, float):  # Filter above if threshold is just one number
            self.DIfilteredDATA = self.polarised_data[self.polarised_data.DI >= self.di_threshold]
        else:  # Filter within an interval if threshold is a tuple or list
            self.DIfilteredDATA = self.polarised_data[(self.di_threshold[0] <= self.polarised_data.DI) & (self.polarised_data.DI <= self.di_threshold[1])]
    
        # Extract relevant data after filtering
        self.DIfilteredGenomes = StringTranspose(self.DIfilteredDATA['diem_genotype'])[1:] # slice off the 'S' column
        self.DIfilteredHIs = [pHetErrOnString(genome) for genome in self.DIfilteredGenomes]
        self.DIfilteredBED = self.DIfilteredDATA[['chrom','start']].values.tolist()
        self.DIpercent = round(100 * len(self.DIfilteredDATA) / len(self.polarised_data))
        self.DIfilteredScafRLEs = RichRLE(self.DIfilteredDATA['chrom'].values.tolist())
        


    def kernel_smooth(self, scale): # ChatGPT drop-in
 #       from collections import defaultdict
        import numpy as np
    
        # --------------------------------------------------
        # 1. Precompute scaffold → indices
        # --------------------------------------------------
        scaffold_indices = defaultdict(list)
        for idx, (scaffold, _) in enumerate(self.DIfilteredBED):
            scaffold_indices[scaffold].append(idx)
    
        # --------------------------------------------------
        # 2. Precompute scaffold → positions array
        # --------------------------------------------------
        scaffold_positions = defaultdict(list)
        for scaffold, pos in self.DIfilteredBED:
            scaffold_positions[scaffold].append(pos)
    
        scaffold_arrays = {
            scaffold: np.asarray(positions)
            for scaffold, positions in scaffold_positions.items()
        }
    
        # --------------------------------------------------
        # 3. Split genomes by scaffold (numeric form)
        # --------------------------------------------------
        # scaffold_haplotypes[scaffold] = list of np.arrays (one per individual)
        scaffold_haplotypes = {
            scaffold: [] for scaffold in scaffold_indices
        }
    
        for genome in self.DIfilteredGenomes:
            for scaffold, indices in scaffold_indices.items():
                # extract once, convert once
                s = ''.join(genome[i] for i in indices)
                s = s.replace("_", "3")
                scaffold_haplotypes[scaffold].append(
                    np.fromiter((ord(c) - 48 for c in s), dtype=np.int8)
                )
    
        # --------------------------------------------------
        # 4. Smooth ALL haplotypes per scaffold (key speedup)
        # --------------------------------------------------
        smoothed_scaffold_haplotypes = {}
    
        for scaffold, haplos in scaffold_haplotypes.items():
            haplo_matrix = np.vstack(haplos)
    
            smoothed = diem.smooth.laplace_smooth_multiple_haplotypes(
                scaffold_arrays[scaffold],
                haplo_matrix,
                scale
            )
    
            smoothed_scaffold_haplotypes[scaffold] = smoothed
    
        # --------------------------------------------------
        # 5. Reassemble genomes (string form)
        # --------------------------------------------------
        n_individuals = len(self.DIfilteredGenomes)
        smoothed_split_genomes = [
            {} for _ in range(n_individuals)
        ]
    
        for scaffold, smoothed_matrix in smoothed_scaffold_haplotypes.items():
            for i in range(n_individuals):
                arr = smoothed_matrix[i]
                chars = np.where(arr == 3, "_", arr.astype(str))
                smoothed_split_genomes[i][scaffold] = ''.join(chars.tolist())
    
        # --------------------------------------------------
        # 6. Finalise
        # --------------------------------------------------
        self.DIfilteredGenomes_unsmoothed = self.DIfilteredGenomes
        self.DIfilteredGenomes = self._reconstruct_genomes(
            smoothed_split_genomes,
            scaffold_indices
        )

    def _reconstruct_genomes(self, smoothed_split_genomes, scaffold_indices):
        reconstructed_genomes = []
    
        for individual in smoothed_split_genomes:
            full_genome = ['0'] * len(self.DIfilteredBED)
    
            for scaffold, indices in scaffold_indices.items():
                scaffold_str = individual[scaffold]
                for i, idx in enumerate(indices):
                    full_genome[idx] = scaffold_str[i]
    
            reconstructed_genome = ''.join(full_genome)
            reconstructed_genomes.append(reconstructed_genome)
    
        return reconstructed_genomes
    
    
    def diem_dithering(self):
        # -------------------------------------------------
        # 1. Group BED entries by chromosome (single pass)
        # -------------------------------------------------
        grouped = defaultdict(list)
        for key, value in self.DIfilteredBED:
            grouped[key].append(value)
    
        chrom_keys = list(grouped.keys())
        self.DIfilteredBED_formatted = [
            np.asarray(grouped[k]) for k in chrom_keys
        ]
        
        # -------------------------------------------------
        # 2. Precompute chromosome spans
        # -------------------------------------------------
        self.length_of_chromosomes = {}
    
        start = 0
        for key, bed_data in zip(chrom_keys, self.DIfilteredBED_formatted):
            end = start + len(bed_data)
            self.length_of_chromosomes[key] = (start, end, len(bed_data))
            start = end
        # -------------------------------------------------
        # 3. Prepare iris_plot_prep ticks (vectorised shift)
        # -------------------------------------------------
        for idx, (key, bed) in enumerate(zip(chrom_keys, self.DIfilteredBED_formatted), start=1):
            x_ticks = fractional_positions_of_multiples(bed, self.ticks)
    
            offset = self.length_of_chromosomes[key][0]
            x_ticks[:, 1] += offset
    
            self.iris_plot_prep[idx] = x_ticks

        # -------------------------------------------------
        # 4. Calculate nBasesDithered SJEB 24 Jan 2026
        # -------------------------------------------------
        ringSpanInBases = 0;
        for chrRefPoses in self.DIfilteredBED_formatted:
            ringSpanInBases = ringSpanInBases + chrRefPoses[-1] - chrRefPoses[0] + 1
        
        # Input argument 'genome_pixels' is number of dithering 'pixels' along genome (pixels may be curved for iris plots)
        # Here, GappedQuotientSplitLengths takes the number of bases that should be dithered together.
        self.nBasesDithered = max(1,round(ringSpanInBases/self.genome_pixels))
        
        # -------------------------------------------------
        # 5. Perform dithering on the filtered data give nBasesDithered
        # -------------------------------------------------
        diem_dit_genomes_bed = [list(group) for _, group in groupby(self.DIfilteredBED, key=lambda x: x[0])]
        processed_diemDITgenomes = []
        for chr in diem_dit_genomes_bed:
            length_data = [row[1] for row in chr]
            split_lengths = self.GappedQuotientSplitLengths(length_data, self.nBasesDithered)# nBasesDithered was self.genome_pixels SJEB 24 Jan 2026 
            processed_diemDITgenomes.append(split_lengths)
        processed_diemDITgenomes = Flatten(processed_diemDITgenomes)
        diemDITgenomes = []
        for genome in self.DIfilteredGenomes:
            string_take_result = StringTakeList(genome, processed_diemDITgenomes)
            state_count = Map(sStateCount, string_take_result)
            combined = list(zip(state_count, processed_diemDITgenomes))
            compressed = self.DITcompress(combined)
            lengths = self.Lengths2StartEnds(compressed)
            diemDITgenomes.append(lengths)
    
        self.diemDITgenomes = diemDITgenomes
    
    
    def generate_plot_label(self, plot_theme):
        """ Generate the label for the plot """
        self.diemPlotLabel = f"{plot_theme} @ DI = {self.di_threshold}: {len(self.DIfilteredDATA)} sites ({self.DIpercent}%) {self.nBasesDithered} bases dithered."
    
    @staticmethod
    def GappedQuotientSplit(lst, Q):
        """
        Splits the list `lst` into sublists where consecutive elements share the same quotient when divided by `Q`.
        """
        quotients = [x // Q for x in lst]
    
        groups = []
        current_group = [lst[0]]
    
        for i in range(1, len(lst)):
            if quotients[i] == quotients[i - 1]:
                current_group.append(lst[i])
            else:
                groups.append(current_group)
                current_group = [lst[i]]
    
        groups.append(current_group)
        return groups
    
    def GappedQuotientSplitLengths(self, lst, Q):
        """
        Returns the lengths of the sublists produced by `gapped_quotient_split`.
        """
        return Map(len, self.GappedQuotientSplit(lst, Q))
    
    @staticmethod
    def normalize_4list(lst):
        """
        Normalizes a 4list by converting each element to its ratio of the total sum.
        Uses Fraction for precise comparison without floating-point errors.
        """
        total = sum(lst)
        if total == 0:
            return tuple(0 for _ in lst)  # Handle case where total is 0
        return tuple(Fraction(x, total) for x in lst)
    
    def DITcompress(self, DITl):
        """
        Compresses the list of {4list, length} tuples.
        """
        grouped_data = [list(group) for _, group in groupby(DITl, key=lambda x: self.normalize_4list(x[0]))]
        final_data = []
        for group in grouped_data:
            summed_states = [sum(x) for x in zip(*(item[0] for item in group))]
            summed_value = sum(item[1] for item in group)
            result = (summed_states, summed_value)
            final_data.append(result)
        return final_data
    
    @staticmethod
    def Lengths2StartEnds(stateNlen):
        lengths = [x[1] for x in stateNlen]
        ends = np.cumsum(lengths)
    
        # Calculate the start positions (end positions minus length plus 1)
        starts = ends - np.array(lengths) + 1
    
        # Combine states, starts, and ends into a list of triplets
        result = [(state, int(start), int(end)) for (state, start, end) in zip([x[0] for x in stateNlen], starts, ends)]
    
        return result
    
def diemPlotPrepFromBedMeta(plot_theme, bed_file_path, meta_file_path,di_threshold,genome_pixels,ticks, smooth = None):

    pzbed, bmIndIDs = read_diem_bed_4_plots(bed_file_path, meta_file_path)

    prep = DiemPlotPrep(
        plot_theme=plot_theme,
        ind_ids=bmIndIDs,
        polarised_data=pzbed,
        di_threshold=di_threshold,
        diemStringPyCol=10,
        di_column=13,
        genome_pixels=genome_pixels,
        ticks=ticks,
        smooth=smooth
    )
    
    return prep    
#________________________________________ END DiemPlotPrep ___________________

#________________________________________ START DiemIris ___________________

import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
# DiemIrisPlot
class WheelDiagram:
    def __init__(self, subplot, center, radius, number_of_rings, cutout_angle=13):
        self.subplot = subplot
        self.center = center
        self.radius = radius
        self.center_radius = radius / 2
        self.number_of_rings = number_of_rings
        self.cutout_angle = cutout_angle
        self.rings_added = 0

    def add_wedge(self, radius, from_angle, to_angle, color):
        self.subplot.add_artist(
            Wedge(self.center, radius, from_angle, to_angle, color=color, clip_on=False) # SJEB this last avoids a world of pain
        )

    def add_ring(self, list_of_thingies):
 #       print(f'Adding ring: {self.rings_added + 1}')
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / list_of_thingies[-1][-1]
        colors = np.array(Map(mcolors.to_rgb,diemColours))

        ring_radius = self.radius - self.rings_added * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(list_of_thingies):
            weights = np.array(thing[0])
            total_weight = np.sum(weights)
            if total_weight == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = np.sum(colors.T * weights, axis=1) / total_weight
            blended_hex = mcolors.to_hex(blended_rgb)
            from_angle = start_angle_offset + 360 - (angle_scale * (thing[1] - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * thing[2])
            self.add_wedge(ring_radius, to_angle,from_angle, blended_hex)

        self.rings_added += 1

    def add_heatmap_ring(self, heatmap):
    # needs work. This version is specific to Honza's MolEcol figures.
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / int(heatmap[-1][-1])
        keys = ["barr", "int", "ovm"]
        values = ["Red", "Blue", "Yellow"]
        color_map = dict(zip(keys, values))

        ring_radius = self.radius + 2 * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(heatmap):
            from_angle = start_angle_offset + 360 - (angle_scale * (int(thing[1]) - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * int(thing[2]))
            self.add_wedge(ring_radius, to_angle, from_angle, color_map[thing[0]])

    def clear_center(self):
        self.add_wedge(self.center_radius, 0, 360, "white")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def diemIrisPlot(
    input_data,
    refposes,
    title=None,
    names=None,
    bed_info=None,
    length_of_chromosomes=None,
    heatmap=None,
):
    # -------------------------------------------------
    # Figure & axes
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    if title is not None:
        ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------------------------------------------------
    # Wheel geometry
    # -------------------------------------------------
    center = np.array((0.5, 0.5))
    radius = 0.48
    cutout_angle = 20
    number_of_rings = len(input_data)

    wd = WheelDiagram(
        ax,
        center,
        radius,
        number_of_rings + (1 if heatmap is not None else 0),
        cutout_angle=cutout_angle,
    )

    if heatmap is not None:
        wd.add_heatmap_ring(heatmap)

    for ring in input_data:
        wd.add_ring(ring)

    wd.clear_center()

    # -------------------------------------------------
    # Geometry helpers (precomputed once)
    # -------------------------------------------------
    available_angle = 360 - cutout_angle
    start_angle_offset = 90
    max_position = input_data[0][-1][2]

    ring_width = (radius - wd.center_radius) / number_of_rings

    chrom_ranges = []
    if length_of_chromosomes is not None:
        for chrom, (start, end, _) in length_of_chromosomes.items():
            chrom_ranges.append(
                (chrom.replace("chromosome_", "Chr "), start, end)
            )

    # -------------------------------------------------
    # Inner chromosome wedges + labels
    # -------------------------------------------------
    if chrom_ranges:
        for idx, (chrom, start, end) in enumerate(chrom_ranges):
            start_angle = start_angle_offset + 360 - available_angle * start / max_position
            end_angle   = start_angle_offset + 360 - available_angle * end   / max_position

            if idx % 2 == 0:
                ax.add_artist(
                    Wedge(center, radius / 2, end_angle, start_angle,
                          color="lightgrey", alpha=0.3)
                )

            midpoint = 0.5 * (start + end)
            mid_angle = start_angle_offset + 360 - available_angle * midpoint / max_position
            mid_rad = np.deg2rad(mid_angle)

            label_xy = center + (radius - 0.28) * np.array(
                [np.cos(mid_rad), np.sin(mid_rad)]
            )

            ax.text(
                label_xy[0],
                label_xy[1],
                chrom,
                ha="center",
                va="center",
                fontsize=8,
                rotation=mid_angle,
                rotation_mode="anchor",
            )

        ax.add_artist(Wedge(center, 0.18, 0, 360, color="white"))

    # -------------------------------------------------
    # Outer ticks
    # -------------------------------------------------
    if bed_info is not None:
        outer_radius = radius + (0.035 if heatmap is not None else 0.015)

        for _, positions in bed_info.items():
            for label, position in positions:
                angle = start_angle_offset + 360 - available_angle * position / max_position
                ang_rad = np.deg2rad(angle)

                base = center + outer_radius * np.array(
                    [np.cos(ang_rad), np.sin(ang_rad)]
                )
                text_pos = base + 0.006 * np.array(
                    [np.cos(ang_rad), np.sin(ang_rad)]
                )

                ax.text(
                    text_pos[0],
                    text_pos[1],
                    str(int(label)),
                    ha="left",
                    va="center",
                    fontsize=6,
                    rotation=angle,
                    rotation_mode="anchor",
                )

                line_start = base - 0.01 * np.array(
                    [np.cos(ang_rad), np.sin(ang_rad)]
                )
                line_end   = line_start + 0.01 * np.array(
                    [np.cos(ang_rad), np.sin(ang_rad)]
                )

                ax.plot(
                    [line_start[0], line_end[0]],
                    [line_start[1], line_end[1]],
                    color="black",
                    linewidth=0.5,
                )

    # -------------------------------------------------
    # Ring (individual) labels
    # -------------------------------------------------
    if names is not None and len(names) == number_of_rings:
        for i, name in enumerate(names):
            ring_radius = radius - (i + 0.5) * ring_width
            label_xy = center + ring_radius * np.array([0, 1])

            ax.text(
                label_xy[0],
                label_xy[1],
                name,
                ha="right",
                va="center",
                fontsize=4,
            )

    # -------------------------------------------------
    # PURE coordinate formatter (browsing)
    # -------------------------------------------------
    def iris_format_coord(x, y):
        # Explicit fallback text (ipympl-safe)
#        fallback = f"(X,Y) = ({x:.3f}, {y:.3f})"
        fallback = " " * 40
    
        dx = x - center[0]
        dy = y - center[1]
        r = np.hypot(dx, dy)
    
        # Outside wheel or inside centre hole
        if r < wd.center_radius or r > radius:
            return fallback
    
        # Angle, clockwise from vertical
        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        rel_angle = (start_angle_offset + 360 - angle) % 360
    
        # Cutout region
        if rel_angle > available_angle:
            return fallback
    
        # Raw genomic position
        raw_genomic_pos = rel_angle / available_angle * max_position
    
        # Chromosome lookup
        chrom_label = None
        for chrom_idx, (chrom, start, end) in enumerate(chrom_ranges):
            if start <= raw_genomic_pos < end:
                chrom_label = chrom
                raw_chrom_fraction = (raw_genomic_pos - start)/(end-start)
#                refpos_idx = round(raw_chrom_fraction * (len(refposes[chrom_idx])-1))
                refpos_idx = int(
                    np.clip(
                        round(raw_chrom_fraction * (len(refposes[chrom_idx]) - 1)),
                        0,
                        len(refposes[chrom_idx]) - 1
                    )
                )
                genomic_pos = refposes[chrom_idx][refpos_idx]
                break
    
        if chrom_label is None:
            return fallback
    
        # Ring (sample) lookup
        ring_idx = int((radius - r) / ring_width)
        if ring_idx < 0 or ring_idx >= number_of_rings:
            return fallback

        if names is None:
            sample = f"ring {ring_idx}"
        else:
            sample = names[ring_idx] 
    
        return f"{chrom_label}   bp={int(genomic_pos):,}   sample={sample}"
    
    ax.format_coord = iris_format_coord

    plt.show()


def diemIrisFromPlotPrep(prepped):
    diemIrisPlot(
    title = prepped.diemPlotLabel,
    input_data = prepped.diemDITgenomes_ordered,
    refposes = prepped.DIfilteredBED_formatted,
    names = prepped.ind_ids,
    bed_info = prepped.iris_plot_prep,
    length_of_chromosomes=prepped.length_of_chromosomes
)
#________________________________________ END DiemIris ___________________

#________________________________________ START diemChromosome ___________________

def diemUnitPlot(
    chromosome_data,
    chromosome_name,
    names_list,
    bed_data,
    path=None,
    ticks=None,
    row_height=10,
):
    """
    Chromosome plot with semantic coordinate display.

    Parameters
    ----------
    chromosome_data : list[str]
        One string per individual (same length).
    index : int
        Scaffold / chromosome index.
    names_list : list[str]
        Individual names (same order as chromosome_data).
    bed_data : np.ndarray
        Genomic positions corresponding to columns.
    ticks : int
        Tick spacing (e.g. 1000).
    row_height : int
        Vertical height per individual.
    """
    # ----------------------------
    # 1. Build numeric grid
    # ----------------------------
    cmap = mcolors.ListedColormap(diemColours)
    grids = []

    #char_to_index = char_to_index

    for genome in chromosome_data:
        index_array = np.fromiter(
            (char_to_index.get(c, 0) for c in genome),
            dtype=int
        )
        grids.append(np.tile(index_array, (row_height, 1)))

    combined_grid = np.vstack(grids)
    n_rows, n_cols = combined_grid.shape

    # ----------------------------
    # 2. Compute ticks
    # ----------------------------
    if ticks is not None:
        x_ticks = fractional_positions_of_multiples(bed_data, ticks)
        x_tick_pos = x_ticks[:, 1]
        x_tick_lab = x_ticks[:, 0].astype(int)
    else:
        x_tick_pos = []
        x_tick_lab = []

    y_tick_pos = np.arange(0, n_rows, row_height)
    y_tick_lab = names_list

    # ----------------------------
    # 3. Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(
        combined_grid,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest'
    )

    # 🔑 Disable image value display ([1.0], [2.0], etc.)
    im.format_cursor_data = lambda z: ""
    
    ax.set_title(chromosome_name)

    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_lab, fontsize=8)

    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(
        x_tick_lab,
        rotation=-90,
        ha='center',
        va='top',
        fontsize=6
    )

    # ----------------------------
    # 4. Custom coordinate display
    # ----------------------------
    def format_coord(x, y):
        col = int(round(x))
        row = int(round(y))

        if not (0 <= col < len(bed_data)):
            return ""

        ind = row // row_height
        if not (0 <= ind < len(y_tick_lab)):
            return ""

        return f"bp={bed_data[col]:,}   sample={y_tick_lab[ind]}"

    ax.format_coord = format_coord

    plt.show()


def diemChromosomeFromPlotPrep(prepped,i):
    diemUnitPlot(
        prepped.unit_plot_prep[i],
        np.array(list(prepped.length_of_chromosomes.keys()))[i],
        bed_data=prepped.DIfilteredBED_formatted[i],
        names_list=prepped.IndIDs_ordered, 
        ticks=1000)
#________________________________________ END DiemChromosome ___________________