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
from matplotlib.patches import Wedge, Polygon, Rectangle
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider
from matplotlib.colors import to_rgb
from matplotlib.colors import LinearSegmentedColormap



# ---- diem internals ----
from . import smooth
from . import contigs as ct


# explicitly used smoothing entry point
from .smooth import laplace_smooth_multiple_haplotypes

# more explicit imports
from fractions import Fraction
from bisect import bisect_left
from joblib import Parallel, delayed # for parallel computation of 'pwdmatrix' pairwise distance matrix rows



""" _____________________ START Mathematica2Python _____________________"""
##############################
#### Mathematica2Python
### Author: Stuart J.E. Baird
###############################


def Split(seq, same_test=lambda a, b: a == b):
    '''
    Function to split a sequence into sublists based on a test function.
    ChatGPT 5.2 provided Mathematica Split equivalent in Python.
    
    Args:
        seq (list): The input sequence to be split.
        same_test (function): A function that takes two arguments and returns True if they are considered the same.
    Returns:
        list: A list of sublists, where each sublist contains consecutive elements that are considered the same.
    '''
    if not seq:
        return []
    out = [[seq[0]]]
    for x in seq[1:]:
        (out[-1] if same_test(out[-1][-1], x) else out.append([x])) and out[-1].append(x)
    return out


def RichRLE(lst):
    """
    Function generating a Rich Run Length Encoding of a list.
    
    Args:
        lst (list): The input list to be encoded.
    Returns:
        lists: A list containing four lists: states, lengths, starts, and ends of each run.
    """
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


def Map(f, lst): 
    """
    equivalent to Mathematica Map 
    """
    return list(map(f, lst))



def ParallelMap(f, lst):
    """ 
    equivalent to Mathematica ParallelMap 
    """
    pool = mp.Pool()
    return list(pool.map(f, lst))


def Flatten(lstOlists): 
    """ 
    equivalent to Mathematica Flatten
    """
    return list(chain.from_iterable(lstOlists)) #itertools



def StringJoin(slst):
    """ 
    equivalent to Mathematica StringJoin 
    """
    separator = ''
    return separator.join(slst)


def Transpose(mat): 
    """ 
    equivalent to Mathematica Transpose
    """
    return list(np.array(mat).T)  # care here - hidden type casting on heterogeneous 'mat'rices


def StringTranspose(slst): 
    """ 
    equivalent to Mathematica StringTranspose 
    """
    return Map(StringJoin, Transpose(Map(Characters, slst)))


def Tally(lst):  # single pass so in principle fast ( O(n) ) but answers unsorted
    """ 
    equivalent to Mathematica Tally 
    """
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

def Second(lst): 
    """ equivalent to Mathematica Second """
    return lst[1]


def Total(lst): 
    """"""
    return sum(lst)


def Join(lst1, lst2): 
    """ equivalent to Mathematica Join """
    return lst1 + lst2


def Take(lst, n):
    """ 
    equivalent to Mathematica Take 
    """
    if n > 0:
        ans = lst[:n]
    elif n == 0:
        ans = lst
    else:
        ans = lst[n:]
    return ans


def Drop(lst, n):
    """ equivalent to Mathematica Drop """
    if n > 0:
        ans = lst[n:]
    elif n == 0:
        ans = lst
    else:
        ans = lst[:n]
    return ans

def FirstPosition(lst, elem):
    """ equivalent to Mathematica FirstPosition """
    i = -1
    pos = []
    for l in lst:
        i += 1
        if l == elem:
            pos.append(i)
            break
    return pos

def Characters(s): 
    """ equivalent to Mathematica Characters """
    return [*s]

def StringTakeList(string, lengths):
    """ equivalent to Mathematica StringTakeList """
    substrings = []
    current_index = 0
    for length in lengths:
        substrings.append(string[current_index:current_index + length])
        current_index += length
    return substrings

""" _____________________ END Mathematica2Python _____________________"""


""" _____________________ START DIEMPy 2023 snippets _____________________"""

##############################
#### From DIEMPy
### Author: Stuart J.E. Baird
###############################


StringReplace20_dict = str.maketrans('02', '20')
""" simultaneous 2<->0 replacement dictionary """

def StringReplace20(text):
    """will _!simultaneously!_ replace 2->0 and 0->2"""
    return text.translate(StringReplace20_dict)


def sStateCount(s):
    """ 
    counts diem States as chars
    Args:   astring
    Returns: list of counts [nU,n0,n1,n2]
    """
    counts = Map(Second, Tally(Join(["0", "1", "2"], Characters(s))))
    nU = Total(
        Drop(counts, 3)
    )  # only the three 'call' chars above are not U encodings!
    counts = list(np.array(Take(counts, 3)) - 1)
    return Join([nU], counts)


def pHetErrOnString(s):
    """
    Calculates state frequency,heterozygosity and error rate from a string of diem states.
    Args:   astring
    Returns: tuple of (pHetErr, pHet, pErr)
    """
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


""" _____________________ START DIEMPy 2023 snippets _____________________"""



""" ______________new support by SJEB STARTING_______________________________"""

def Chr_Nickname(chr_name):
    """
    Shorten chromosome names for plotting.
    E.g., 'chromosome_1' -> 'Chr 1'
    Args:
        chr_name (str): Full chromosome name.
    Returns:
        str: Shortened chromosome name.
    """
    if 'scaffold' in chr_name:
        return chr_name.replace('scaffold_', 'Scaf ')[-7:]
    elif 'chromosome' in chr_name:
        return chr_name.replace('chromosome_', 'Chr ')[-6:]
    else:
        return chr_name[-7:]
    
def Ind_Nickname(ind_name):
    """
    Shorten Ind names for plotting.
    E.g., 'chromosome_1' -> 'Chr 1'
    Args:
        ind_name (str): Full individual name.
    Returns:
        str: Shortened individual name.
    """
    if '_NA' in ind_name:
        return ind_name.replace('_NA', ' NA')[-10:]
    else:
        return ind_name[-10:]
    

def read_diem_bed_4_plots(bed_file_path, meta_file_path):
    """
    Reads a diem BED file and meta file for use in plots.py
    code copy from Derek Setter's read_diem_bed with additions by SJEB
    
    Derek comments:
    Fast version of read_diem_bed with significant performance improvements.
    
    Args:
    bed_file_path (str): Path to the diem BED file.
    meta_file_path (str): Path to the diem metadata file.

    Returns:
    A tuple of a DiemType object containing the diem BED data 
    (POLARISED (if hasPolarity)) and an IndsName file.
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


def genomes_summary_given_DI(aDT, DIthreshold: float):
    """
    Summarises a diemType.DMBC across chromosomes, applying a DI threshold filter.
    ChatGPT 5.2 speed optimised version.

    Args:
    aDT: a DiemType
    DIthreshold: DI threshold filter
    Returns:
    summaries: list of per-individual summary arrays [HI, HOM1, HET, HOM2, U]
    RetainedNumer: number of retained sites after DI filtering
    RetainedDenom: total number of sites before DI filtering
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
    Summarises a diemType.DMBC for a chromosomes, applying a DI threshold filter.

    Args:
    aDT: a DiemType
    DIthreshold: DI threshold filter
    chrom_idx: index of chromosome to process
    Returns:
    summaries: list of per-individual summary arrays [HI, HOM1, HET, HOM2, U]
    RetainedNumer: number of retained sites after DI filtering
    RetainedDenom: total number of sites before DI filtering
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

def statewise_genomes_summary_given_DI(aDT, DIthreshold: float):
    """
    Statewise summary of genomes under a DI threshold.

    Refinements over genomes_summary_given_DI:
      1) counts3 = count of states NOT in {0,1,2}
      2) returns per-chromosome per-individual state counts
      3) returns per-chromosome retained counts

    Args
    ----
    aDT : DiemType
        Must provide:
          - DMBC : list of arrays, each shape (nInds, nSites_chr)
          - DIByChr : list of 1D arrays, per chromosome
          - chrPloidies : list of per-individual ploidies
    DIthreshold : float
        DI filter threshold

    Returns
    -------
    chrom_counts : list of dicts, length nChr
        Each dict has keys:
          'counts0', 'counts1', 'counts2', 'counts3'
        Each value is a float array of shape (nInds,)
        Counts are ploidy-weighted.
    chrom_retained : list of tuples, length nChr
        Each element is (RetainedNumer_chr, RetainedDenom_chr)
    """

    nChr = len(aDT.DMBC)
    nInds = aDT.DMBC[0].shape[0]

    chrom_counts = []
    chrom_retained = []

    for chr_idx in range(nChr):
        SM = aDT.DMBC[chr_idx]               # (nInds, nSites)
        DI = aDT.DIByChr[chr_idx]            # (nSites,)
        ploidies = aDT.chrPloidies[chr_idx]  # (nInds,)

        DIfilter = DI >= DIthreshold

        RetainedNumer = int(np.count_nonzero(DIfilter))
        RetainedDenom = int(DIfilter.size)
        chrom_retained.append((RetainedNumer, RetainedDenom))

        if RetainedNumer == 0:
            chrom_counts.append({
                "counts0": np.zeros(nInds, dtype=float),
                "counts1": np.zeros(nInds, dtype=float),
                "counts2": np.zeros(nInds, dtype=float),
                "counts3": np.zeros(nInds, dtype=float),
            })
            continue

        SMf = SM[:, DIfilter]  # (nInds, nRetained)

        # Vectorised state masks
        is0 = (SMf == 0)
        is1 = (SMf == 1)
        is2 = (SMf == 2)

        # counts3 = NOT in {0,1,2}
        is3 = ~(is0 | is1 | is2)

        # Per-individual counts
        counts0 = is0.sum(axis=1)
        counts1 = is1.sum(axis=1)
        counts2 = is2.sum(axis=1)
        counts3 = is3.sum(axis=1)

        # Apply ploidy weights once
        w = ploidies.astype(float)

        chrom_counts.append({
            "counts0": w * counts0,
            "counts1": w * counts1,
            "counts2": w * counts2,
            "counts3": w * counts3,
        })

    return chrom_counts, chrom_retained

def summaries_from_statewise_counts(chrom_counts):
    """
    Compute [HI, HOM1, HET, HOM2, U] from statewise chrom_counts.

    chrom_counts: iterable of dicts with keys
        'counts0', 'counts1', 'counts2', 'counts3'
        (arrays of shape nInds)

    Returns:
        summaries = [HI, HOM1, HET, HOM2, U]
    """

    A0 = sum(c["counts0"] for c in chrom_counts)
    A1 = sum(c["counts1"] for c in chrom_counts)
    A2 = sum(c["counts2"] for c in chrom_counts)
    A3 = sum(c["counts3"] for c in chrom_counts)

    dipDenom   = A1 + A2 + A3
    hapDenom   = 2 * dipDenom
    stateDenom = dipDenom + A0

    HI   = (A2 + 2 * A3) / hapDenom
    HOM1 = A1 / dipDenom
    HET  = A2 / dipDenom
    HOM2 = A3 / dipDenom
    U    = A0 / stateDenom

    return [HI, HOM1, HET, HOM2, U]


def genomes_strings_given_DI(dPol, DIthreshold, *, flatten=True, as_array=True):
    """
    Efficiently return DI-filtered genome strings or arrays.

    Assumes:
      - dPol.DMBC : list of arrays (n_individuals, n_sites_chr)
      - dPol.DIByChr : list of DI arrays per chromosome
      - dPol.DIfilteredGenomes semantics already validated
    """

    n_inds = dPol.DMBC[0].shape[0]
    filtered_chunks = []

    for chr_idx, DM in enumerate(dPol.DMBC):
        DIfilter = dPol.DIByChr[chr_idx] >= DIthreshold
        if not np.any(DIfilter):
            continue

        # slice once
        SMf = DM[:, DIfilter]

        # convert to characters efficiently
        # assume states {0,1,2,3} map to '_','0','1','2' or similar
        # adjust mapping if needed
        state_map = np.array(['_', '0', '1', '2'], dtype='<U1')
        filtered_chunks.append(state_map[SMf])

    if not filtered_chunks:
        if as_array:
            return np.empty((n_inds, 0), dtype='<U1')
        return [''] * n_inds

    G = np.concatenate(filtered_chunks, axis=1)

    if as_array:
        return G

    # string form
    return [''.join(row) for row in G]


def fractional_positions_of_multiples(A, delta):
    """
    Calculate fractional positions of multiples of delta in a sorted array A.

    Used as an inverse linear interpolation from reference (physical) positions to site indices;
    for example, to place ticks at regular physical intervals on a plot of site indices.

    Args:
        A (array-like): Sorted array of values (e.g. physical positions of DI filtered SNVs).
        delta (float): The interval for multiples (e.g. 1000 for kb ticks).
    Returns:
        np.ndarray: Array of (value (tick label), position (tick placement on SNV metric)) pairs.
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


"""________________________________________ START GenomeSummariesPlot ___________________"""

class GenomeSummaryPlot:
    """
    Plots genome summaries with DI filtering and interactive widgets.

    These summaries include HI, HOM1, HET, HOM2, and U proportions per individual.
    Cursor hover displays individual IDs.
    Reorder button sorts individuals by HI given current DI filter.
    
    Args:
        dPol: DiemType object containing genomic data.
    """
    def __init__(self, dPol):
        self.dPol = dPol

        # ---- initial state ----
        self.IndNickNames = [Ind_Nickname(name) for name in dPol.indNames]
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
            np.arange(len(self.IndNickNames)),
            np.array(self.IndNickNames)[self.indHIorder],
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
            label='IndLabels font',
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
        #self.summaries, self.DInumer, self.DIdenom = genomes_summary_given_DI(
        #    self.dPol, val
        #)
        self.chrom_counts, self.chrom_retained = \
            statewise_genomes_summary_given_DI(self.dPol, val)

        self.summaries = summaries_from_statewise_counts(self.chrom_counts)

        self.DInumer = sum(n for n, _ in self.chrom_retained)
        self.DIdenom = sum(d for _, d in self.chrom_retained)
        # END of statewise insert
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


class GenomeMultiSummaryPlot:
    """
    Plots genome summaries per chromosome with DI filtering and interactive widgets.
    These summaries include HI, HOM1, HET, HOM2, and U proportions per individual.
    Cursor hover displays individual IDs.
    Reorder button sorts individuals by global HI given current DI filter.
    Args:
        dPol: DiemType object containing genomic data.
        chrom_indices: List of chromosome indices to plot.
    """
    def __init__(self, dPol, chrom_indices, max_cols=3):
        self.dPol = dPol
        self.IndNickNames = [Ind_Nickname(name) for name in dPol.indNames]  
        self.max_cols = max_cols

        # ---- validate chromosomes ----
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)

        # ---- ordering state ----
        self.indNameFont = 6

        self.chrom_counts, self.chrom_retained = \
            statewise_genomes_summary_given_DI(self.dPol, float("-inf"))
        # build per-chromosome summaries once (initial DI)
        self.chrom_summaries = {}
        for idx in self.chrom_indices:
            self.chrom_summaries[idx] = summaries_from_statewise_counts(
                [self.chrom_counts[idx]]
    )
        
        # authoritative whole-genome summaries (from statewise counts)
        global_summaries = summaries_from_statewise_counts(self.chrom_counts)
        self.global_HI = global_summaries[0]
        global_hi_colour = "cyan"
        self.indHIorder = np.argsort(self.global_HI)
        # END statewise insert


        # ---- grid layout ----
        n_plots = len(self.chrom_indices)
        n_cols = min(self.max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

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
            bottom=0.45,# <-- important: room for sliders.
            hspace=0.60,   # ← increase this for vert gap between
            wspace=0.25
        )

        # ---- draw plots ----
        self.lines = {}
        axes_flat = self.axes.flatten()

        colours = Flatten(
            [['red'], diemColours[1:], ['gray']]
        )

        self.chrom_axes = {}
        self.global_hi_lines = {}

        for ax, chrom_idx in zip(axes_flat, self.chrom_indices):
            self.chrom_axes[chrom_idx] = ax
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

            # add global HI reference (purple)
            global_hi_line, = ax.plot(
                self.global_HI[self.indHIorder],
                color=global_hi_colour,
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
            )

            self.global_hi_lines[chrom_idx] = global_hi_line


            ax.set_ylim(0, 1)
            num, denom = self.chrom_retained[chrom_idx]
            ax.set_title(
                f"Chr {chrom_idx} | {num:,}/{denom:,} sites",
                fontsize=10
            )
            ax.tick_params(axis='x', rotation=55)

            ax.set_xticks(
                np.arange(len(self.IndNickNames)),
                np.array(self.IndNickNames)[self.indHIorder],
                fontsize=self.indNameFont,
                ha='right'
            )

        # hide unused axes
        for ax in axes_flat[len(self.chrom_indices):]:
            ax.axis("off")

        # legend once
        axes_flat[0].legend(
            ['HIc', 'HOM1', 'HET', 'HOM2', 'U', 'HIg'],
            fontsize=8,
            frameon=False
        )

        # ---- widgets ----
        self._init_widgets()

        # ---- coordinate display ----
        self._install_format_coord()

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
    # Helper
    # ==================================================

    def _install_format_coord(self):
        """
        Install per-axis coordinate display showing IndID when
        cursor is close to a plotted point.
        """

        n = len(self.dPol.indNames)
        tolerance = 0.03  # vertical proximity in y-units

        axes_flat = self.axes.flatten()[:len(self.chrom_indices)]

        for ax, chrom_idx in zip(axes_flat, self.chrom_indices):
            chrom_lines = self.lines[chrom_idx]

            def make_format_coord(ax, chrom_lines):
                def format_coord(x, y):
                    fallback = "\u2007" * 30

                    i = int(round(x))
                    if i < 0 or i >= n:
                        return fallback

                    for line in chrom_lines:
                        ydata = line.get_ydata()
                        if abs(y - ydata[i]) < tolerance:
                            return f"IndID: {self.dPol.indNames[self.indHIorder[i]]}"

                    return fallback

                return format_coord

            ax.format_coord = make_format_coord(ax, chrom_lines)

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
        ax_FS = self.fig.add_axes([0.25, 0.12, 0.1, 0.03])
        self.FONT_slider = Slider(
            ax_FS,
            "IndLabel font",
            4, 16,
            valinit=self.indNameFont,
            valstep=1
        )
        self.FONT_slider.on_changed(self._on_font_change)

        # reorder button
        ax_RE = self.fig.add_axes([0.75, 0.115, 0.15, 0.045])
        self.reorder_button = Button(
            ax_RE,
            "Reorder by global HI",
            hovercolor="0.95",
            color="cyan" # should be same as global HI line color
        )
        self.reorder_button.on_clicked(self._on_reorder)

    # ==================================================
    # Callbacks
    # ==================================================

    def _on_DI_change(self, val):
        # recompute statewise counts
        self.chrom_counts, self.chrom_retained = \
            statewise_genomes_summary_given_DI(self.dPol, val)

        # recompute per-chromosome summaries (cheap)
        for idx in self.chrom_indices:
            summaries = summaries_from_statewise_counts(
                [self.chrom_counts[idx]]
            )
            self.chrom_summaries[idx] = summaries
            # recompute global HI from statewise counts
            global_summaries = summaries_from_statewise_counts(self.chrom_counts)
            self.global_HI = global_summaries[0]

            # update plotted data
            for line, summary in zip(self.lines[idx], summaries):
                line.set_ydata(summary[self.indHIorder])

            # update global HI overlay
            self.global_hi_lines[idx].set_ydata(
                self.global_HI[self.indHIorder]
            )

            # update title with retained counts
            num, denom = self.chrom_retained[idx]
            self.chrom_axes[idx].set_title(
                f"Chr {idx} | {num:,}/{denom:,} sites",
                fontsize=10
            )

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

        # update global HI ordering
        for idx in self.chrom_indices:
            self.global_hi_lines[idx].set_ydata(
                self.global_HI[self.indHIorder]
            )
    
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

"""________________________________________ END GenomeMultiSummaryPlot ___________________"""


"""________________________________________ START GenomicDeFinettiPlot ___________________"""


class GenomicDeFinettiPlot:
    """
    Plots a genomic de Finetti plot with DI filtering and interactive widgets.
    Cursor hover displays individual IDs.

    c.f. 
    Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044

    Figure 4:
    Ebdon, S., Laetsch, D. R., Vila, R., Baird, S. J. E., & Lohse, K. (2025). 
    Genomic regions of current low hybridisation mark long-term barriers to gene flow 
    in scarce swallowtail butterflies. PLoS Genetics, 21(4), 30. 
    doi:https://doi.org/10.1371/journal.pgen.1011655

    Uses genomes_summary_given_DI

    Args:
        dPol: DiemType object containing genomic data.
    """
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
    # EURG
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
        ax_SZ = self.fig.add_axes([0.25, 0.10, 0.1, 0.03])
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

"""________________________________________ END GenomicDeFinettiPlot ___________________"""


"""________________________________________ START GenomicMultiDeFinettiPlot ___________________"""

class GenomicMultiDeFinettiPlot:
    """
    Multiple de Finetti plots, one per chromosome,
    all controlled by a shared DI slider and size slider.

    Uses statewise_genomes_summary_given_DI

    c.f. 
    Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044
    
    Figure 4:
    Ebdon, S., Laetsch, D. R., Vila, R., Baird, S. J. E., & Lohse, K. (2025). 
    Genomic regions of current low hybridisation mark long-term barriers to gene flow 
    in scarce swallowtail butterflies. PLoS Genetics, 21(4), 30. 
    doi:https://doi.org/10.1371/journal.pgen.1011655

    Args:
        dPol: DiemType object containing genomic data.
        chrom_indices: List of chromosome indices to plot.
    """

    def __init__(self, dPol, chrom_indices, max_cols=3):
        self.dPol = dPol
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)
        self.max_cols = max_cols

        self.marker_size = 60
        self.n_ind = len(dPol.indNames)
        self.indHIorder = np.arange(self.n_ind)

        # ---------- initial statewise computation ----------
        self.chrom_counts, self.chrom_retained = \
            statewise_genomes_summary_given_DI(self.dPol, float("-inf"))

        # global summaries (authoritative ordering)
        self.global_summaries = summaries_from_statewise_counts(self.chrom_counts)
        self.global_HI = self.global_summaries[0]
        self.indHIorder = np.argsort(self.global_HI)

        # ---------- layout ----------
        n_plots = len(self.chrom_indices)
        n_cols = min(self.max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        self.fig, self.axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.8 * n_cols, 4.6 * n_rows),
            squeeze=False
        )

        self.fig.subplots_adjust(
            left=0.06, right=0.98,
            top=0.92, bottom=0.32,
            hspace=0.45, wspace=0.25
        )

        # ---------- draw ----------
        self.scatters = {}
        self.chrom_axes = {}

        axes_flat = self.axes.flatten()

        for ax, idx in zip(axes_flat, self.chrom_indices):
            self.chrom_axes[idx] = ax
            self._setup_axes(ax)
            self._draw_triangle(ax)
            self._draw_hwe_curve(ax)

            summaries = summaries_from_statewise_counts(
                [self.chrom_counts[idx]]
            )
            _, HOM1, HET, HOM2, U = summaries

            x, y = self._to_triangle_coords(
                HOM1[self.indHIorder],
                HET[self.indHIorder],
                HOM2[self.indHIorder]
            )

            colours = self._blend_colours(HOM1, HET, HOM2, U)

            sc = ax.scatter(
                x, y,
                s=self.marker_size,
                c=colours[self.indHIorder],
                edgecolor="black",
                linewidth=0.3
            )

            num, denom = self.chrom_retained[idx]
            ax.set_title(f"Chr {idx} | {num:,}/{denom:,} sites", fontsize=10)

            self.scatters[idx] = sc

        for ax in axes_flat[len(self.chrom_indices):]:
            ax.axis("off")

        # ---------- widgets ----------
        self._init_widgets()

        # ---------- hover ----------
        self._install_format_coord()

        plt.show()

    # ======================================================
    # Helpers
    # ======================================================

    @staticmethod
    def _to_triangle_coords(hom1, het, hom2):
        x = hom2 + 0.5 * het
        y = (np.sqrt(3) / 2) * het
        return x, y

    def _blend_colours(self, HOM1, HET, HOM2, U):
        weights = np.column_stack([HOM1, HET, HOM2, U])
        base = np.array([
            to_rgb(diemColours[1]),
            to_rgb(diemColours[2]),
            to_rgb(diemColours[3]),
            to_rgb(diemColours[0]),
        ])
        return np.clip(weights @ base, 0, 1)

    def _setup_axes(self, ax):
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def _draw_triangle(self, ax):
        h = np.sqrt(3)/2
        ax.add_patch(Polygon([[0,0],[1,0],[0.5,h]], fill=False, lw=1.2))

        ax.text(0, -0.04, "HOM1", ha="center", va="top", fontsize=8)
        ax.text(1, -0.04, "HOM2", ha="center", va="top", fontsize=8)
        ax.text(0.5, h + 0.03, "HET", ha="center", va="bottom", fontsize=8)

    def _draw_hwe_curve(self, ax):
        p = np.linspace(0,1,400)
        x,y = self._to_triangle_coords(p*p, 2*p*(1-p), (1-p)**2)
        ax.plot(x,y,color="black",lw=0.8,alpha=0.5)

    # ======================================================
    # Widgets
    # ======================================================

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        ax_DI = self.fig.add_axes([0.15, 0.20, 0.70, 0.035])
        self.DI_slider = Slider(ax_DI, "DI", *DI_span, valinit=DI_span[0])
        self.DI_slider.on_changed(self._on_DI_change)

        ax_SZ = self.fig.add_axes([0.25, 0.16, 0.1, 0.03])
        self.size_slider = Slider(ax_SZ, "Symbol size", 10, 300, valinit=self.marker_size)
        self.size_slider.on_changed(self._on_size_change)

    # ======================================================
    # Callbacks
    # ======================================================

    def _on_DI_change(self, val):
        self.chrom_counts, self.chrom_retained = \
            statewise_genomes_summary_given_DI(self.dPol, val)

        self.global_summaries = summaries_from_statewise_counts(self.chrom_counts)
        self.indHIorder = np.argsort(self.global_summaries[0])

        for idx in self.chrom_indices:
            summaries = summaries_from_statewise_counts(
                [self.chrom_counts[idx]]
            )
            _, H1, Ht, H2, U = summaries

            x,y = self._to_triangle_coords(
                H1[self.indHIorder],
                Ht[self.indHIorder],
                H2[self.indHIorder]
            )

            sc = self.scatters[idx]
            sc.set_offsets(np.column_stack([x,y]))
            sc.set_facecolors(
                self._blend_colours(H1,Ht,H2,U)[self.indHIorder]
            )

            num, denom = self.chrom_retained[idx]
            self.chrom_axes[idx].set_title(
                f"Chr {idx} | {num:,}/{denom:,} sites", fontsize=10
            )

        self.fig.canvas.draw_idle()

    def _on_size_change(self, val):
        self.marker_size = int(val)
        for sc in self.scatters.values():
            sc.set_sizes(np.full(self.n_ind, self.marker_size))
        self.fig.canvas.draw_idle()

    # ======================================================
    # Hover
    # ======================================================

    def _install_format_coord(self):
        tol = 0.03
        names = self.dPol.indNames

        for idx, sc in self.scatters.items():
            ax = self.chrom_axes[idx]

            def make_fmt(sc):
                def fmt(x,y):
                    pts = sc.get_offsets()
                    d = np.hypot(pts[:,0]-x, pts[:,1]-y)
                    i = np.argmin(d)
                    if d[i] < tol:
                        return f"IndID: {names[self.indHIorder[i]]}"
                    return "\u2007"*30
                return fmt

            ax.format_coord = make_fmt(sc)

    # ======================================================
    # Validation
    # ======================================================

    def _validate_chrom_indices(self, chrom_indices):
        max_idx = len(self.dPol.DMBC) - 1
        valid = [i for i in chrom_indices if 0 <= int(i) <= max_idx]
        if not valid:
            raise ValueError("No valid chromosome indices")
        return valid



"""________________________________________ END GenomicMultiDeFinettiPlot ___________________"""


"""________________________________________ START GenomicContributionsPlot ___________________"""

class GenomicContributionsPlot:
    """
    Plots per-chromosome genomic contributions (HOM1, HET, HOM2, U, excluded)
    with DI filtering and interactive widgets.

    uses  statewise_genome_summary_given_DI

    Args:
        dPol: DiemType object containing genomic data.
    """
    def __init__(self, dPol):
        self.dPol = dPol
        self.fontsize = 8

        # initial computation
        self.DInumer = 0
        self.DIdenom = 0
        self._compute_contributions(float("-inf"))

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.format_coord = None
        self.fig.subplots_adjust(bottom=0.40, right=0.85)
     
        self._draw_bars()
        self._init_widgets()

        self.ax.set_title('Genomic Contributions; no DI filter')

        plt.show()

    # --------------------------------------------------
    # Core computation
    # --------------------------------------------------


    def _compute_contributions(self, DIval):
        chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(
            self.dPol, DIval
        )

        n_chr = len(chrom_counts)
        self.DInumer = 0
        self.DIdenom = 0

        self.chrom_labels = []
        self.props = np.zeros((n_chr, 5))  # HOM1, HET, HOM2, U, excluded

        for i in range(n_chr):
            chr_name = Chr_Nickname(self.dPol.chrNames[i])
            self.chrom_labels.append(chr_name)

            counts = chrom_counts[i]
            kept_sites, total_sites = chrom_retained[i]
            self.DInumer = self.DInumer + kept_sites
            self.DIdenom = self.DIdenom + total_sites

            # ---- summed ploidy-weighted counts over individuals ----
            c0 = np.sum(counts["counts0"])
            c1 = np.sum(counts["counts1"])
            c2 = np.sum(counts["counts2"])
            c3 = np.sum(counts["counts3"])

            total_alleles = total_sites * np.sum(self.dPol.chrPloidies[i])  # SJEB was bugged. ChatGPT unaware of ploidy
            if total_alleles == 0:
                continue


            self.props[i, :] = [
                c1 / total_alleles,                                  # HOM1
                c2 / total_alleles,                                  # HET
                c3 / total_alleles,                                  # HOM2
                c0 / total_alleles,                                  # U
                (1.0 - kept_sites / total_sites) if total_sites else 0 # excluded
            ]

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

        #self.ax.set_title(
        #    "Genomic contributions by chromosome  DI ≥ {:.2f}".format(self.current_DI),
        #    pad=12
        #)  EURG
        prop = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0

        self.ax.set_title(
            "Genomic de Finetti plot  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop),
            fontsize=12,
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
        ax_FS = self.fig.add_axes([0.15, 0.13, 0.15, 0.03])
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




"""________________________________________ END GenomicContributions__________________"""





"""________________________________________ START diemPaisPlot ___________________"""

def pwmatrixFromDiemType(aDT, DIthreshold=float("-inf")):
    """
    Compute a DI-filtered pairwise distance matrix from a DiemType object.

    Args:
        aDT : DiemType
        DIthreshold : float
            Only sites with DI >= DIthreshold are retained.

    Returns:
        M : (N, N) numpy array
            Symmetric pairwise distance matrix.
    """

    # -------------------------------------------------
    # Dimensions
    # -------------------------------------------------
    n_ind = aDT.DMBC[0].shape[0]

    # -------------------------------------------------
    # Pairwise weight matrix (codes 0..3)
    # -------------------------------------------------
    W = np.zeros((4, 4), dtype=float)

    W[1, 2] = W[2, 1] = 1
    W[1, 3] = W[3, 1] = 2
    W[2, 2] = 1
    W[2, 3] = W[3, 2] = 1
    # (1,1) and (3,3) remain 0

    # -------------------------------------------------
    # Accumulators
    # -------------------------------------------------
    num = np.zeros((n_ind, n_ind), dtype=float)
    den = np.zeros((n_ind, n_ind), dtype=float)

    # -------------------------------------------------
    # Single pass over chromosomes
    # -------------------------------------------------
    for chr_idx, SM in enumerate(aDT.DMBC):
        # SM shape: (n_ind, n_sites)
        DIvals = aDT.DIByChr[chr_idx]
        keep = DIvals >= DIthreshold

        if not np.any(keep):
            continue

        SMf = SM[:, keep]

        # iterate retained sites
        for s in range(SMf.shape[1]):
            col = SMf[:, s]

            valid = col != 0
            idx = np.where(valid)[0]

            if idx.size < 2:
                continue

            vals = col[idx]

            # pairwise contribution
            for ii, i in enumerate(idx):
                ai = vals[ii]
                for jj in range(ii + 1, len(idx)):
                    j = idx[jj]
                    aj = vals[jj]
                    w = W[ai, aj]

                    num[i, j] += w
                    num[j, i] += w
                    den[i, j] += 1
                    den[j, i] += 1

    # -------------------------------------------------
    # Final matrix
    # -------------------------------------------------
    M = np.full((n_ind, n_ind), np.nan)
    mask = den > 0
    M[mask] = num[mask] / den[mask]
    #np.fill_diagonal(M, 0.0) an example of ChatGPT 5.2 suggestion that is not biologically valid

    return M



# for parallel computation of pairwise distance matrix rows
def _pwmatrix_row(i, G, W):
    """
    Compute one row of the pairwise distance matrix.
    """
    n = G.shape[0]
    row = np.zeros(n, dtype=float)

    ai = G[i]
    for j in range(n):
        aj = G[j]
        valid = (ai != 0) & (aj != 0)
        denom = valid.sum()
        if denom == 0:
            row[j] = np.nan
        else:
            row[j] = W[ai[valid], aj[valid]].sum() / denom

    return i, row


def PARApwmatrixFromDiemType(
    aDT,
    DIthreshold=float("-inf"),
    n_jobs=-1,
    backend="loky",
):
    """
    Parallel computation of pairwise distance matrix.

    Args
    ----
    aDT : DiemType
    DIthreshold : float
    n_jobs : int
        Number of cores (-1 = all)
    backend : str
        joblib backend ("loky" recommended)

    Returns
    -------
    M : (n_ind, n_ind) numpy array
    """

    # -------------------------------------------------
    # DI-filtered genomes (ONCE)
    # -------------------------------------------------
    G_chars = genomes_strings_given_DI(aDT, DIthreshold, as_array=True)
    if G_chars.shape[1] == 0:
        n = G_chars.shape[0]
        return np.full((n, n), np.nan)

    # map chars → codes {_,0,1,2} → {0,1,2,3}
    G = np.zeros(G_chars.shape, dtype=np.int8)
    G[G_chars == "0"] = 1
    G[G_chars == "1"] = 2
    G[G_chars == "2"] = 3

    n = G.shape[0]

    # -------------------------------------------------
    # Distance weight matrix
    # -------------------------------------------------
    W = np.zeros((4, 4), dtype=float)
    W[1, 2] = W[2, 1] = 1
    W[1, 3] = W[3, 1] = 2
    W[2, 2] = 1
    W[2, 3] = W[3, 2] = 1

    # -------------------------------------------------
    # Parallel row computation
    # -------------------------------------------------
    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        prefer="processes",
        batch_size=1
    )(
        delayed(_pwmatrix_row)(i, G, W)
        for i in range(n)
    )

    # -------------------------------------------------
    # Assemble matrix
    # -------------------------------------------------
    M = np.zeros((n, n), dtype=float)
    for i, row in results:
        M[i, :] = row

    return M


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


class diemPairsPlot:
    """
    Pairwise distance plot using brickDiagram semantics.

    Uses genomes_summary_given_DI and PARApwmatrixFromDiemType

    c.f. Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044

    Coding co-pilot: ChatGPT 5.2

    Left panel:
        Square heatmap of pairwise distances (brick rectangles),
        ordered by Hybrid Index at the specified DI threshold.

    Right panel:
        Vertical colour key.

    Hover:
        Shows "IndA × IndB : distance".
    """

    def __init__(self, dPol, DIthreshold=float("-inf"), figsize=(9, 6)):
        self.dPol = dPol
        self.DIthreshold = DIthreshold

        # -------------------------------------------------
        # Compute HI ordering (authoritative)
        # -------------------------------------------------
        summaries, _, _ = genomes_summary_given_DI(dPol, DIthreshold)
        HI = summaries[0]
        self.ind_order = np.argsort(HI)

        self.indNames = np.array(dPol.indNames)[self.ind_order]
        self.n = len(self.indNames)

        # -------------------------------------------------
        # Compute pairwise matrix
        # -------------------------------------------------
        self.M = PARApwmatrixFromDiemType(
            dPol,
            DIthreshold=DIthreshold
            #n_jobs=-1 
        )

        # apply ordering once
        self.M = self.M[self.ind_order][:, self.ind_order]

        # -------------------------------------------------
        # Figure layout
        # -------------------------------------------------
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[20, 1],
            wspace=0.08
        )

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.cax = self.fig.add_subplot(gs[0, 1])

        # -------------------------------------------------
        # Colormap
        # -------------------------------------------------
        self.cmap = LinearSegmentedColormap.from_list(
            "soft_coolwarm",
            ["#1e90ff", "white", "#fff266", "#ff1a1a"]
        )

        self.vmin = np.nanmin(self.M)
        self.vmax = np.nanmax(self.M)

        # -------------------------------------------------
        # Draw heatmap as bricks
        # -------------------------------------------------
        self._draw_bricks()

        # -------------------------------------------------
        # Axes formatting
        # -------------------------------------------------
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(0, self.n)
        self.ax.set_aspect("equal")

        centers = np.arange(self.n) + 0.5
        self.ax.set_xticks(centers)
        self.ax.set_yticks(centers)
        self.ax.set_xticklabels(self.indNames, rotation=90, fontsize=7)
        self.ax.set_yticklabels(self.indNames, fontsize=7)

        self.ax.set_title(
            f"Pairwise distances (DI ≥ {DIthreshold:.2f})",
            pad=10
        )

        # -------------------------------------------------
        # Colour key
        # -------------------------------------------------
        self._draw_colour_key()
        # -------------------------------------------------
        # Font slider
        # -------------------------------------------------
        self._init_font_slider()
        # -------------------------------------------------
        # Hover browsing
        # -------------------------------------------------
        self._install_format_coord()

        plt.show()

    # =================================================
    # Font size slider
    # =================================================

    def _init_font_slider(self):
        # create a small axis directly beneath the colour key
        pos = self.cax.get_position()
        slider_height = 0.035

        ax_fs = self.fig.add_axes([
            pos.x0,
            pos.y0 - slider_height - 0.015,
            pos.width,
            slider_height
        ])

        self.label_fontsize = 7

        self.font_slider = Slider(
            ax_fs,
            "Labels",
            4,
            16,
            valinit=self.label_fontsize,
            valstep=1
        )

        self.font_slider.on_changed(self._on_fontsize_change)

    # =================================================
    # Drawing
    # =================================================

    def _draw_bricks(self):
        norm = plt.Normalize(self.vmin, self.vmax)

        for i in range(self.n):
            for j in range(self.n):
                val = self.M[j, i]
                if not np.isfinite(val):
                    color = "black"
                else:
                    color = self.cmap(norm(val))

                rect = Rectangle(
                    (i, j),
                    1, 1,
                    facecolor=color,
                    edgecolor="none"
                )
                self.ax.add_patch(rect)

    def _draw_colour_key(self):
        gradient = np.linspace(self.vmin, self.vmax, 256).reshape(-1, 1)
        self.cax.imshow(
            gradient,
            aspect="auto",
            cmap=self.cmap,
            origin="lower"
        )
        self.cax.set_yticks([0, 255])
        self.cax.set_yticklabels(
            [f"{self.vmin:.2f}", f"{self.vmax:.2f}"],
            fontsize=8
        )
        self.cax.set_xticks([])
        self.cax.set_title("Distance", fontsize=9)

    # =================================================
    # Call backs
    # =================================================

    def _on_fontsize_change(self, val):
        fs = int(val)
        self.label_fontsize = fs

        for lbl in self.ax.get_xticklabels():
            lbl.set_fontsize(fs)
        for lbl in self.ax.get_yticklabels():
            lbl.set_fontsize(fs)

        self.fig.canvas.draw_idle()
    # =================================================
    # Hover logic
    # =================================================

    def _install_format_coord(self):
        n = self.n

        def format_coord(x, y):
            fallback = " " * 40

            i = int(np.floor(x))
            j = int(np.floor(y))

            if 0 <= i < n and 0 <= j < n:
                a = self.indNames[j]
                b = self.indNames[i]
                d = self.M[j, i]
                if np.isfinite(d):
                    return f"{a} × {b} : {d:.3f}"
                return f"{a} × {b} : NA"

            return fallback

        self.ax.format_coord = format_coord

"""________________________________________ END diemPairsPlot ___________________"""




"""________________________________________ START DiemPlotPrep__________________"""

class DiemPlotPrep:
    """ 
    Prepares data for DI-based plotting, including filtering, smoothing, dithering, and label generation.
    Args:
        plot_theme: Theme for plotting.
        ind_ids: List of individual IDs.
        polarised_data: DataFrame containing polarised genomic data.
        di_threshold: DI threshold for filtering.
        di_column: Column name for DI values.
        diemStringPyCol: Column name for Diem genotype strings.
        genome_pixels: Number of genome pixels for dithering.
        ticks: Optional ticks for plotting.
        smooth: Optional smoothing parameter.
    """
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
"""________________________________________ END DiemPlotPrep ___________________"""



"""________________________________________ START DiemIris ___________________"""

class WheelDiagram:
    """
    Utility class for creating wheel diagrams (iris plots).
    Args:
        subplot: Matplotlib subplot to draw on.
        center: Center coordinates of the wheel.
        radius: Outer radius of the wheel.
        number_of_rings: Number of concentric rings in the wheel.
        cutout_angle: Angle of the cutout section (default is 13 degrees).
    """
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


"""________________________________________ END WheelDiagram ___________________"""

"""________________________________________ START diemIrisPlot ___________________"""

def diemIrisPlot(
    input_data,
    refposes,
    title=None,
    names=None,
    bed_info=None,
    length_of_chromosomes=None,
    heatmap=None,
):
    """
    Creates an iris plot (wheel diagram) based on the provided input data.
    Args:
        input_data: List of rings, where each ring is a list of triplets (weights, start_pos, end_pos).
        refposes: Reference positions for the data.
        title: Optional title for the plot.
        names: Optional list of names for each ring.
        bed_info: Optional dictionary with BED information for outer ticks.
        length_of_chromosomes: Optional dictionary with chromosome lengths for inner wedges.
        heatmap: Optional heatmap data for an additional ring.
    Returns:
        Matplotlib figure and axes objects.
    """
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
                #(chrom.replace("chromosome_", "Chr "), start, end)# Nickname Chromosomes
                (Chr_Nickname(chrom), start, end)# Use Nickname function
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
"""________________________________________ END DiemIris ___________________"""


"""________________________________________ START diemLongPlot ___________________"""



class BrickDiagram:
    """
    Utility class for creating linear (brick) genome diagrams.

    Each ring is a horizontal band.
    Each brick spans a genomic interval [start, end).
    """

    def __init__(self, subplot, x_min, x_max, y_min, y_max, number_of_rings):
        self.subplot = subplot
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.number_of_rings = number_of_rings
        self.rings_added = 0

        self.ring_height = (y_max - y_min) / number_of_rings

    def add_brick(self, x0, x1, ring_idx, color):
        y0 = self.y_max - (ring_idx + 1) * self.ring_height
        width = x1 - x0

        rect = Rectangle(
            (x0, y0),
            width,
            self.ring_height,
            facecolor=color,
            edgecolor=None,
            linewidth=0,
            clip_on=False
        )
        self.subplot.add_patch(rect)

    def add_ring(self, list_of_thingies, colors):
        """
        list_of_thingies: [(weights, start_pos, end_pos), ...]
        colors: RGB base colours (same as iris)
        """
        ring_idx = self.rings_added

        for thing in list_of_thingies:
            weights, start, end = thing
            weights = np.asarray(weights)

            total = weights.sum()
            if total == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = (colors.T * weights).sum(axis=1) / total

            self.add_brick(start, end, ring_idx, mcolors.to_hex(blended_rgb))

        self.rings_added += 1





class BrickDiagram:
    """
    Draws horizontal rings made of rectangles spanning [x0,x1) in data coords.
    """
    def __init__(self, ax, n_rings, y_min=0.0, y_max=1.0):
        self.ax = ax
        self.n_rings = n_rings
        self.y_min = y_min
        self.y_max = y_max
        self.ring_h = (y_max - y_min) / n_rings

    def add_brick(self, x0, x1, ring_idx, color):
        y0 = self.y_max - (ring_idx + 1) * self.ring_h
        self.ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                self.ring_h,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                clip_on=False
            )
        )


def diemLongPlot(
    input_data,
    refposes,
    chrom_indices,
    title=None,
    names=None,
    length_of_chromosomes=None,
):
    """
    Linear analogue of diemIrisPlot.

    IMPORTANT SEMANTICS:
      - input_data ring intervals (start,end) are GLOBAL positions along the concatenated genome.
      - length_of_chromosomes defines GLOBAL [start,end) ranges for each chromosome in that same coordinate system.
      - chrom_indices selects which chromosomes to plot, and we pack them contiguously with no gaps.
    """
    if length_of_chromosomes is None:
        raise ValueError("diemLongPlot: length_of_chromosomes is required for chromosome selection/packing.")

    # -------------------------------------------------
    # Build stable chromosome list in the same order used elsewhere
    # -------------------------------------------------
    chrom_items = list(length_of_chromosomes.items())  # preserves insertion order

    max_idx = len(chrom_items) - 1
    chrom_indices = [int(i) for i in chrom_indices if 0 <= int(i) <= max_idx]
    if not chrom_indices:
        raise ValueError("diemLongPlot: chrom_indices contained no valid indices.")

    # Selected chromosome global ranges and packed ranges
    # store: (orig_idx, label, g0, g1, p0, p1)
    selected = []
    packed_cursor = 0.0

    for orig_idx in chrom_indices:
        chrom_name, v = chrom_items[orig_idx]
        #g0, g1 = _unpack_chrom_range(v)
        # robust unpack: supports (start,end) or (start,end,length)
        if len(v) == 2:
            g0, g1 = v
        elif len(v) >= 3:
            g0, g1 = v[0], v[1]
        else:
            raise ValueError(f"Unexpected chromosome range value: {v!r}")

        g_len = g1 - g0
        if g_len <= 0:
            continue

        label = Chr_Nickname(chrom_name)  # your existing helper
        p0 = packed_cursor
        p1 = packed_cursor + g_len
        selected.append((orig_idx, label, float(g0), float(g1), float(p0), float(p1)))
        packed_cursor = p1

    if not selected:
        raise ValueError("diemLongPlot: selected chromosomes have no positive lengths.")

    packed_len = selected[-1][5]

    # -------------------------------------------------
    # Figure & axes
    # -------------------------------------------------
    n_rings = len(input_data)
    fig, ax = plt.subplots(figsize=(11, 4))  # <- GenomeSummaryPlot-like ratio
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.18,
        top=0.92,
    )

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, packed_len)
    ax.set_ylim(-0.9, n_rings)  # room for chromosome bricks + labels

    ax.set_aspect("auto")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------------------------------------------------
    # Draw rings as bricks (global->packed, with intersection)
    # -------------------------------------------------
    # colours as in iris
    colors = np.array(list(map(mcolors.to_rgb, diemColours)))
    bd = BrickDiagram(ax, n_rings, y_min=0, y_max=n_rings)

    for ring_idx, ring in enumerate(input_data):
        for weights, g_start, g_end in ring:
            g_start = float(g_start-1)  # convert to 0-based SJEB 24 Jan 2026
            g_end   = float(g_end)
            if g_end <= g_start:
                continue

            w = np.asarray(weights)
            tot = float(np.sum(w))
            if tot == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = (colors.T * w).sum(axis=1) / tot

            # Intersect this global interval with each selected chrom global range,
            # then map overlap into packed coords.
            for (orig_idx, _, cg0, cg1, cp0, _) in selected:
                o0 = max(g_start, cg0)
                o1 = min(g_end,   cg1)
                if o1 <= o0:
                    continue

                # packed x: cp0 + (global - cg0)
                x0 = cp0 + (o0 - cg0)
                x1 = cp0 + (o1 - cg0)
                bd.add_brick(x0, x1, ring_idx, mcolors.to_hex(blended_rgb))

    # -------------------------------------------------
    # Chromosome bricks + labels at base (no gaps)
    # -------------------------------------------------
    base_y = -0.62 
    #base_h = 0.45 * 30

    for i, (_, label, _, _, p0, p1) in enumerate(selected):
        if i % 2 == 0:
            ax.axvspan(p0,p1,color="grey", alpha=0.35,ymin=-0.18,ymax=0.01, clip_on=False)


        ax.text(
            0.5 * (p0 + p1),
            base_y - 0.12,
            label,
            ha="center",
            va="top",
            fontsize=8,
            rotation=90
        )

    # -------------------------------------------------
    # Ring labels (optional)
    # -------------------------------------------------
    if names is not None and len(names) == n_rings:
        for i, name in enumerate(names):
            y = n_rings - i - 0.5
            ax.text(
                -0.01 * packed_len,
                y,
                name,
                ha="right",
                va="center",
                fontsize=6,
                clip_on=False
            )

    # -------------------------------------------------
    # Cursor browsing (packed -> chrom -> refpose)
    # -------------------------------------------------
    def long_format_coord(x, y):
        fallback = " " * 40

        # ring index from y
        ring_idx = int(np.floor(n_rings - y))
        if ring_idx < 0 or ring_idx >= n_rings:
            return fallback

        # locate chromosome in packed coords
        chrom_hit = None
        for (orig_idx, label, cg0, cg1, p0, p1) in selected:
            if p0 <= x < p1:
                chrom_hit = (orig_idx, label, cg0, cg1, p0, p1)
                break
        if chrom_hit is None:
            return fallback

        orig_idx, chrom_label, cg0, cg1, p0, p1 = chrom_hit

        # position within chromosome (global coordinate)
        chrom_len = cg1 - cg0
        if chrom_len <= 0:
            return fallback

        frac = (x - p0) / (p1 - p0)  # in [0,1)
        # map to refposes for that original chromosome index
        ref = refposes[orig_idx]
        if len(ref) == 0:
            return fallback

        ref_i = int(np.clip(round(frac * (len(ref) - 1)), 0, len(ref) - 1))
        genomic_pos = ref[ref_i]

        sample = names[ring_idx] if (names is not None and len(names) == n_rings) else f"ring {ring_idx}"
        return f"{chrom_label}   bp={int(genomic_pos):,}   sample={sample}"

    ax.format_coord = long_format_coord

    plt.show()

def diemLongFromPlotPrep(prepped, chrom_indices):
    diemLongPlot(
        title=prepped.diemPlotLabel,
        input_data=prepped.diemDITgenomes_ordered,
        refposes=prepped.DIfilteredBED_formatted,
        names=prepped.ind_ids,
        length_of_chromosomes=prepped.length_of_chromosomes,
        chrom_indices=chrom_indices,
    ) 


"""________________________________________ END diemLongPlot ___________________"""

