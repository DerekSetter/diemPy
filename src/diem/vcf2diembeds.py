#!/usr/bin/env python3

"""vcf2diembeds.py

Usage
  vcf_file_path = 'path/to/your/file.vcf'
  vcf2diembed.py vcf_file_path

Options:
  None
 
Features:
  vcf files of any size (eg 26Gb filesize, 474 genomes, 3.6 million sites, 37 mins (see increment...now 10 mins?)
  lossless encoding of vcf variants of up to 4 states (diptypes with further states are U-encoded)
  outputs 3 bed-like files (TSV with #-commented header)
       1 file.vcf.diem_input.bed      variant input for diem
       2 file.vcf.diem_exclude.bed    vcf sites excluded for diem
       3 file.vcf.diem_meta.bed       default metadata input for diem

  The new release of diem.py should run on {1,3}
  Ideally 3 will first be updated by the user before running diem
  (using 'update_diem_metadata' in prep)

Development: (Stuart J.E. Baird)
    2. increment: extended from '0123456789' encoding to DIEM alphabet (55 symbols; 0–9,a–z,A–S)
       This means 'main' has some stuff before the bit that resembles the original chatGPT5 vanilla VCF interpreter
    1. increment: Optimised file I/O (4x speedup).
   
    0. Inspired by vcf2diem.py by Sam Ebdon this is, however, a complete re-write.
    I asked chatGPT5 to write me a vanilla vcf interpreter (here, ± == main)
    I then added 5 functions (immediately below) and tidied.
    The result is much shorter and simpler than vcf2diem.py, with fewer dependencies.
    This is to be expected, as it does a different job: here there is minimal filtering.
    In the emBEDed diem workflow filtering experiments should be downstream of the [slow: big files] VCF interpreter.
"""

import sys, os
from timeit import default_timer as timer
from collections import Counter
from datetime import timedelta
import io, gzip
import operator
import itertools  # used by diem_most_common_alleles (downstream tool)


def open_text(path):
    if path.endswith(".gz"):
        # For reading, compresslevel is ignored; use TextIOWrapper for decoding
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", buffering=1 << 20, newline="")

def open_out(path):
    return open(path, "w", encoding="utf-8", buffering=1 << 20, newline="")

def convert_seconds(n):
    return str(timedelta(seconds=n))

#---------------- DIEM alphabet (55 symbols) start -------------------------#
diemALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS'
diemUencodableChar = '_'
diemMaxVariants = 10
diemMaxChars = diemMaxVariants * (diemMaxVariants + 1) // 2  # 55

def diemGentobase62(n):
    if 0 <= n < diemMaxChars:
        return diemALPHABET[n]
    else:
        return '!'  # Error case

# used once to create a lookup table
def base62ordConversion(n):
    if 48 <= n <= 57:
        return n - 48  # '0'..'9' -> 0..9
    elif 65 <= n <= 90:
        return n - 29  # 'A'..'Z' -> 36..61 (we only use up to 'S')
    elif 97 <= n <= 122:
        return n - 87  # 'a'..'z' -> 10..35
    else:
        return -1  # Error case

# a lookup tuple: ord(char) -> diemGen index (or -1)
base62todiemGenLookUp = tuple(base62ordConversion(n) for n in range(123))
def base62todiemGen(s):
    if len(s) == 1:
        n = ord(s)
        if 0 <= n <= 122:
            return base62todiemGenLookUp[n]
    return -1  # Error case
   
# two small lookup tuples ensure compatibility with 0123456789 encoding without runtime cost
ABSijLookupTable = (0, 0, 5, 5, 5, 9, 11, 13, 15, 17)
MAXijLookupTable = (0, 0, 0, 0, 3, 6, 10, 15, 21, 28)
def diem_encode_allele_ranks(i, j):
    maxij = max(i, j)
    if maxij >= diemMaxVariants:
        return diemUencodableChar
    absij = abs(i - j)
    return diemGentobase62(i + j + ABSijLookupTable[absij] + MAXijLookupTable[maxij])  # see lossless encoding slideset

# construct decode lookup
diem_decode_allele_ranks_lookup = [[base62todiemGen(diem_encode_allele_ranks(i, j)), [i, j]]
                                   for j in range(diemMaxVariants) for i in range(diemMaxVariants) if i <= j]
diem_decode_allele_ranks_lookup.sort(key=lambda x: x[0])
diem_decode_allele_ranks_lookup[:] = [row[1] for row in diem_decode_allele_ranks_lookup]

def diem_decode_allele_ranks(s):
    diemgen = base62todiemGen(s)
    if 0 <= diemgen < diemMaxChars:
        return diem_decode_allele_ranks_lookup[diemgen]
    else:
        return []

def diem_most_common_alleles(site_string):  # tool for downstream
    gtypes = [diem_decode_allele_ranks(c) for c in site_string]
    c = Counter(list(itertools.chain(*gtypes)))
    return c.most_common()
#---------------- DIEM alphabet end ---------------------------#

exclusion_NOT = 'E0'   # NOT excluded
exclusion_invar = 'E1' # invariant is simply excluded    

def diem_encode_HOMsingleton_exclusion(diem_gtypes: str) -> str:
    """O(1) string ops: counts & first indices; no per-char lists."""
    lz = diem_gtypes.count('0')
    lt = diem_gtypes.count('2')
    if lz > 1 and lt > 1:
        return exclusion_NOT
    if lz == 0 and lt == 0:
        return 'E2'  # ZERO HOMs
    if lz == 1 and lt == 1:
        zi = diem_gtypes.find('0')
        ti = diem_gtypes.find('2')
        return f'E5_{zi}_{ti}'  # TWO SINGLETON HOMS
    if lz == 0 or lt == 0:
        return 'E3'  # ONLY ONE HOM
    # One singleton HOM case
    idx = diem_gtypes.find('0') if lz == 1 else diem_gtypes.find('2')
    return f'E4_{idx}'

def fastGTs(sample_fields, seqLabelsString, gt_index):
    """Hot path: derive DIEM string & metadata directly from sample_fields."""
    hide_exclusions = False  # internal flag
    diem_enc_a_ranks = diem_encode_allele_ranks
    diem_dec_a_ranks = diem_decode_allele_ranks
   
    seqLabels = seqLabelsString.split(',')
    GTlabelcount = [0] * diemMaxVariants

    def iter_GTs():
        """Yield GT strings with minimal splitting/allocation."""
        if gt_index == 0:
            for s in sample_fields:
                yield s.split(':', 1)[0] if s else './.'
        elif gt_index > 0:
            for s in sample_fields:
                # split up to gt_index to avoid trailing allocations
                parts = s.split(':', gt_index)
                yield parts[gt_index] if len(parts) > gt_index else './.'
        else:
            for _ in sample_fields:
                yield './.'

    def GT(s):
        # normalize phased to unphased once
        if '|' in s:
            s = s.replace('|', '/')
        match s:
            case '0/0':
                GTlabelcount[0] += 2; return '0'
            case '0/1':
                GTlabelcount[0] += 1; GTlabelcount[1] += 1; return '1'
            case '1/1':
                GTlabelcount[1] += 2; return '2'
            case _:
                a, _, b = s.partition('/')
                if a == '.' or not b:
                    return diemUencodableChar
                i = int(a); j = int(b)
                if max(i, j) >= diemMaxVariants:
                    return diemUencodableChar
                GTlabelcount[i] += 1; GTlabelcount[j] += 1
                return diem_enc_a_ranks(i, j)

    prelim_states = [GT(g) for g in iter_GTs()]

    # ---- Early return: invariant / mono-allelic -> skip sorting & translation
    nonzero = diemMaxVariants - GTlabelcount.count(0)
    if nonzero < 2:
        SNV = 0
        if nonzero == 0:
            usedSeqs = [diemUencodableChar]
        else:
            usedSeqs = [seqLabels[0]]  # only the most-common allele is present
        return nonzero, ','.join(usedSeqs), SNV, exclusion_invar, 'S'
    # ----

    simpleOrder = list(range(diemMaxVariants))
    # top-N variant labels by count
    sorted_count_pos = sorted(
        zip(GTlabelcount, simpleOrder), key=lambda x: x[0], reverse=True
    )[:nonzero]
    sorted_count_pos = [pos for _, pos in sorted_count_pos]

    ztExchangeableOrder = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9][:nonzero]

    if (sorted_count_pos == simpleOrder[:nonzero]
        or sorted_count_pos == ztExchangeableOrder):  # NO reordering necessary
        usedSeqs = seqLabels[:nonzero]
        SNV = int(max(map(len, usedSeqs)) == 1)
        diemString = 'S' + ''.join(prelim_states)
        exclusion_crit = diem_encode_HOMsingleton_exclusion(diemString)
    else:  # reordering possibly necessary (not checking tie possibilities), rare
        getter = operator.itemgetter(*sorted_count_pos)
        picked = getter(seqLabels)
        usedSeqs = list(picked) if isinstance(picked, tuple) else [picked]
        SNV = int(max(map(len, usedSeqs)) == 1)

        if nonzero == 1:
            diemString = 'S'
            exclusion_crit = exclusion_invar
        else:
            new_ranks = getter(simpleOrder)
            if not isinstance(new_ranks, tuple):
                new_ranks = (new_ranks,)
            revhash = [-1] * (max(new_ranks) + 1)
            for i, pos in enumerate(new_ranks):
                revhash[pos] = i

            oldstates = set(prelim_states)
            oldrankpairs = [[s, diem_dec_a_ranks(s)] for s in oldstates if s != diemUencodableChar]
            replacements = [(s, diem_enc_a_ranks(revhash[rp[0]], revhash[rp[1]]))
                            for s, rp in oldrankpairs]

            prelimdiemString = 'S' + ''.join(prelim_states)
            chr_translation = str.maketrans(dict(replacements))
            diemString = prelimdiemString.translate(chr_translation)
            exclusion_crit = diem_encode_HOMsingleton_exclusion(diemString)

    if exclusion_crit != exclusion_NOT and hide_exclusions:  # hide excluded sites' strings
        diemString = 'S'

    return nonzero, ','.join(usedSeqs), SNV, exclusion_crit, diemString


def main():
    file_path = sys.argv[1]
    start_time = timer()

    variant_output_path = file_path + '.diem_input.bed'
    exclude_output_path = file_path + '.diem_exclude.bed'
    meta_output_path    = file_path + '.diem_meta.bed'

    samples = []
    chromosome_tally = Counter()
    minpos = {}   # chrom -> min start
    maxpos = {}   # chrom -> max end
    nvars = necls = 0

    with open_text(file_path) as vcf, \
         open_out(variant_output_path) as vout, \
         open_out(exclude_output_path) as xout:

        invariant_header = '#Chrom\tStart\tEnd\tQual\tRef\tSeqAlleles\tSNV\tnVNTs\tExclusionCriterion\t'
        # headers
        vout.write(invariant_header + '|'.join([]) + '\n')  # will patch later after samples known
        xout.write(invariant_header + 'Dstring\n')

        gt_index = None
        last_fmt = None

        for line in vcf:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                header_fields = line.rstrip('\n').split('\t')
                samples = header_fields[9:]
                # rewrite variant header now that we know samples
                vout.seek(0)
                vout.write(invariant_header + '|'.join(samples) + '\n')
                vout.seek(0, io.SEEK_END)
                continue

            fields = line.rstrip('\n').split('\t')
            chrom = fields[0]
            pos   = fields[1]
            start = str(int(pos) - 1)   # VCF 1-based -> BED 0-based
            end   = pos                 # single-site interval description
            ref   = fields[3]
            alt   = fields[4]
            qual  = fields[5]
            fmt   = fields[8]
            sample_fields = fields[9:]

            # compute GT column index once per site-format (handle changing FORMAT)
            if gt_index is None or fmt != last_fmt:
                fmt_cols = fmt.split(':')
                gt_index = fmt_cols.index('GT') if 'GT' in fmt_cols else -1
                last_fmt = fmt

            # Interpret GTs fast (no intermediate genotypes list)
            nVNTs, ordered_SAs, SNV, exclusion_criterion, dg = fastGTs(sample_fields, ref + ',' + alt, gt_index)
           
            if exclusion_criterion == exclusion_NOT:
                vout.write(f"{chrom}\t{start}\t{end}\t{qual}\t{ref}\t{ordered_SAs}\t{SNV}\t{nVNTs}\t{exclusion_criterion}\t{dg}\n")
                nvars += 1
                chromosome_tally[chrom] += 1
                if chrom not in minpos or int(start) < int(minpos[chrom]): minpos[chrom] = start
                if chrom not in maxpos or int(end)   > int(maxpos[chrom]): maxpos[chrom] = end
            else:
                xout.write(f"{chrom}\t{start}\t{end}\t{qual}\t{ref}\t{ordered_SAs}\t{SNV}\t{nVNTs}\t{exclusion_criterion}\t{dg}\n")
                necls += 1

    # Write META (once, buffered)
    default_ploidy = ['2'] * len(samples)
    with open_out(meta_output_path) as mout:
        mout.write('#Chrom\tStart_diem_input\tEnd_diem_input\tn(diem_inputs)\t' + '\t'.join(samples) + '\n')
        for chrom, count in chromosome_tally.items():
            mout.write(f"{chrom}\t{minpos[chrom]}\t{maxpos[chrom]}\t{count}\t" + '\t'.join(default_ploidy) + '\n')

    total = nvars + necls
    print(f"{total:,d} vcf sites for {len(samples)} genomes, split to {nvars:,d} diem variants with {necls:,d} excluded. ({round(100 * necls / max(total,1))}% reduction).")
    print(f"diem_meta.bed data saved to {meta_output_path}")
    print(f"diem_input.bed data saved to {variant_output_path}")
    print(f"diem_exclude.bed data saved to {exclude_output_path}")
    print(f"Time taken (hours:mins:secs): {convert_seconds(timer() - start_time)}")

if __name__ == "__main__":
    main()