# --- diem2fasta: clean notebook-friendly rewrite ---

"""
diem2fasta

Convert vcf2diem outputs (diem_input.bed + optional diem_exclude.bed) into:
- FASTA reconstructions per reference scaffold (or per requested regions),
- per-sample heterozygosity summary,
- pairwise distance matrices computed from DIEM site-state strings (Dstrings).

Core assumptions / semantics:

Dstrings
- Each row describes one site; the Dstring begins with 'S' then one character per sample
  in the META/VCF sample order.
- The DIEM alphabet encodes diplotypes formed from pairs of the ten most common alleles at a site.
- Unencodable state: any site-state character that is '_' or 'U', or any character not
  decodable by the DIEM decode table. This includes:
    (i) diplotypes involving alleles outside the ten most common, and
    (ii) missing/no-data.
- Unencodable calls do not contribute to distance coverage.

Reference / coordinates
- Reference scaffold names and lengths are taken from the provided reference FASTA.
- META is treated as auxiliary metadata (e.g., sample order/ploidy) and is not used to
  define what regions are converted to FASTA.
- If regions_bed_path is provided, diem2fasta writes ONLY the region FASTA(s) described
  there (labels preserved); otherwise it writes full-length per-scaffold FASTAs.

Distances
- Unphased Hamming with partial credit: distance is computed per site from decoded diplotypes,
  accounting for unknown phase; unencodable calls are skipped (coverage counts computed sites).
- Compatibility distance: distance 0 means the pair could be consistent with the same underlying
  sequence given missing/unencodable calls; conflicts are scored from decoded diplotypes.
"""


from __future__ import annotations
import os, sys, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pysam
from multiprocessing import Pool, cpu_count

# Treat BOTH '_' and 'U' as standard “unencodable”, plus anything unknown.

def build_diem_decoder():
    """
    Build DIEM char -> (i,j) map for i<=j for the 10-allele model.
    Returns:
      decode: dict[str, tuple[int,int] | None]
      is_unencodable_char: callable(ch)->bool, treating '_'/'U'/unknown as unencodable
    """
    diemALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS'
    diemMaxVariants = 10
    diemMaxChars = diemMaxVariants * (diemMaxVariants + 1) // 2
    ABSij = (0, 0, 5, 5, 5, 9, 11, 13, 15, 17)
    MAXij = (0, 0, 0, 0, 3, 6, 10, 15, 21, 28)

    def enc(i, j):
        if max(i, j) >= diemMaxVariants:
            return None
        n = i + j + ABSij[abs(i - j)] + MAXij[max(i, j)]
        if 0 <= n < diemMaxChars:
            return diemALPHABET[n]
        return None

    decode = {}
    for j in range(diemMaxVariants):
        for i in range(j + 1):
            ch = enc(i, j)
            if ch is not None:
                decode[ch] = (i, j)

    # Standard “unencodable” sentinels
    decode["_"] = None
    decode["U"] = None

    UNENC = {"_", "U"}

    def is_unencodable_char(ch: str) -> bool:
        # Anything unknown OR sentinel is unencodable.
        # (Unknown chars: decode.get -> None)
        return (ch in UNENC) or (decode.get(ch, None) is None)

    return decode, is_unencodable_char

    
# -------------------- IUPAC helpers --------------------

IUPAC_HET = {
    frozenset(("A","G")): "R",
    frozenset(("C","T")): "Y",
    frozenset(("G","C")): "S",
    frozenset(("A","T")): "W",
    frozenset(("G","T")): "K",
    frozenset(("A","C")): "M",
}
def iupac_from_pair(a: str, b: str) -> str:
    """Return IUPAC for unordered (a,b) DNA; identical -> a; unknown -> 'N'."""
    if a == b:
        return a
    return IUPAC_HET.get(frozenset((a,b)), "N")

# -------------------- IO: meta / bed --------------------

def load_meta(meta_path: str,
              ref_fasta_path: str,
              variants_path: str | None = None):
    """
    META reader for diem2fasta:

    • If META has RefStart0/RefEnd0 -> use them for lengths (new format).
    • Otherwise (old META with Start_diem_input/End_diem_input) ignore those for coords
      and infer 0..len(FASTA[chrom]).
    • META is still used for scaffold list/order (#Chrom) and sample columns/order if present.
    • If META lacks samples, infer sample order from the last header column of the variants BED
      (pipe-joined names).
    """
    import pandas as pd, numpy as np, pysam

    # --- FASTA is the fallback (and the truth for old META)
    fa = pysam.FastaFile(ref_fasta_path)
    fa_order = list(fa.references)
    fa_len   = {r: fa.get_reference_length(r) for r in fa_order}

    # --- Read META
    df = pd.read_csv(meta_path, sep="\t", dtype="string")

    # #Chrom (case tolerant)
    chrom_col = None
    for c in df.columns:
        if c.lower() in ("#chrom", "chrom"):
            chrom_col = c; break
    if chrom_col is not None:
        df = df.rename(columns={chrom_col: "#Chrom"})
        chr_names = df["#Chrom"].astype(str).to_numpy()
    else:
        # fallback if META doesn't list chroms
        if variants_path:
            # minimal scan to get a chrom column from variants
            nskip = 0
            with open(variants_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("##"): nskip += 1; continue
                    if line.startswith("#"):  nskip += 1
                    break
            dv = pd.read_csv(variants_path, sep="\t", dtype="string", engine="python",
                             nrows=200000, skiprows=max(nskip-1, 0))
            chrom_alt = None
            for c in dv.columns:
                if c.lower() in ("#chrom", "chrom"):
                    chrom_alt = c; break
            if chrom_alt is not None:
                chr_names = np.array(sorted(dv[chrom_alt].dropna().astype(str).unique()), dtype=object)
            else:
                chr_names = np.array(fa_order, dtype=object)
        else:
            chr_names = np.array(fa_order, dtype=object)

    # --- Decide how to compute lengths
    # Prefer explicit RefStart0/RefEnd0 if present; otherwise infer from FASTA.
    refstart = next((c for c in df.columns if c.lower() == "refstart0"), None)
    refend   = next((c for c in df.columns if c.lower() == "refend0"),   None)

    if refstart and refend:
        # Newer META: trust these values
        s = pd.to_numeric(df[refstart], errors="coerce").fillna(0).astype("int64").to_numpy()
        e = pd.to_numeric(df[refend],   errors="coerce").to_numpy()
        # align lengths to chr_names order
        len_map = {}
        for i in range(len(df)):
            c = str(df.loc[i, "#Chrom"]) if "#Chrom" in df.columns else None
            if c is not None and pd.notna(e[i]):
                len_map[c] = int(e[i]) - int(s[i])
        chr_lengths = np.array([int(len_map.get(str(c), fa_len.get(str(c), 0)))
                                for c in chr_names], dtype="int64")
    else:
        # Old META: ignore Start_diem_input/End_diem_input; use reference lengths
        chr_lengths = np.array([int(fa_len.get(str(c), 0)) for c in chr_names], dtype="int64")

    # --- Samples & ploidy
    # Heuristic: everything to the right of the last known fixed meta field is samples.
    fixed_candidates = [
        "n(diem_inputs)", "lastvariantpos0", "firstvariantstart0",
        "end_diem_input", "start_diem_input", "end"
    ]
    low = {c.lower(): c for c in df.columns}
    last_fixed_idx = -1
    for key in fixed_candidates:
        if key in low:
            last_fixed_idx = max(last_fixed_idx, list(df.columns).index(low[key]))

    if last_fixed_idx >= 0 and last_fixed_idx + 1 < len(df.columns):
        sample_names = df.columns[last_fixed_idx+1:].to_numpy()
        # per-chrom ploidy, default to 2 where missing
        ploidy = df.iloc[:, last_fixed_idx+1:].astype("Int64").fillna(2).to_numpy()
        # align ploidy to chr_names if #Chrom present
        if "#Chrom" in df.columns and len(ploidy) == len(df):
            row_map = {str(df.loc[i, "#Chrom"]): i for i in range(len(df))}
            ploidy = np.vstack([
                (ploidy[row_map[c]] if c in row_map else np.full(len(sample_names), 2, dtype=int))
                for c in map(str, chr_names)
            ])
    else:
        # No sample block in META → infer from variants header (pipe-joined)
        if not variants_path:
            raise ValueError("Cannot infer sample names: META has no sample columns and variants_path not provided.")
        last_header = None
        with open(variants_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#Chrom") or line.startswith("#CHROM"):
                    last_header = line.rstrip("\n").split("\t")[-1]
                    break
        if last_header is None:
            raise ValueError("Could not find variants header to infer sample names.")
        sample_names = np.array(last_header.split("|"), dtype=object)
        ploidy = np.full((len(chr_names), len(sample_names)), 2, dtype="int64")

    return chr_names, chr_lengths, sample_names, ploidy

def read_regions_bed(path):
    """
    Read BED of regions to export.

    Columns:
        chrom  start  end  [label...]

    The 4th column may contain spaces — everything after column 3 is
    treated as the label verbatim.

    Returns list of tuples:
        (chrom:str, start:int, end:int, label:str)
    """
    regions = []

    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")

            if not line or line.startswith("#"):
                continue

            # Split only first 3 columns
            parts = line.split(maxsplit=3)

            if len(parts) < 3:
                raise ValueError(f"{path}:{ln} — fewer than 3 columns")

            chrom = parts[0]
            start = int(parts[1])
            end   = int(parts[2])

            # Label = remainder verbatim if present
            if len(parts) == 4:
                label = parts[3]
            else:
                label = f"{chrom}:{start}-{end}"

            regions.append((chrom, start, end, label))

    return regions


def read_bed_as_strings(path: str, skip_preamble: bool=True
                       ) -> pd.DataFrame:
    """
    Read a DIEM BED-like TSV as all strings. The last column is the Dstring
    and is renamed to 'diem_genotype'. Works for both variants and excludes.
    """
    # Count preamble (##...) and header (#Chrom...) lines
    nskip = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                nskip += 1
                continue
            if line.startswith("#"):
                nskip += 1
            break

    df = pd.read_csv(path, sep="\t", dtype="string", engine="python", skiprows=nskip-1)
    # last column contains D-string with very long header (pipe-joined samples)
    lastcol = df.columns[-1]
    df = df.rename(columns={lastcol: "diem_genotype"})
    # normalize key columns
    for c in ("chrom","Chrom","#Chrom"):
        if c in df.columns:
            df = df.rename(columns={c: "chrom"})
            break
    if "exclusion_criterion" not in df.columns:
        for c in df.columns:
            if c.lower() == "exclusioncriterion":
                df = df.rename(columns={c:"exclusion_criterion"})
                break
    return df

def union_variants(df_v: pd.DataFrame, df_x: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Keep all rows; excludes are useful except pure invariants E1."""
    if df_x is None:
        return df_v.copy()
    df_all = pd.concat([df_v, df_x], ignore_index=True)
    # normalize exclusion codes and drop E1
    if "exclusion_criterion" in df_all.columns:
        df_all["exclusion_criterion"] = df_all["exclusion_criterion"].astype("string").str.strip()
        df_all = df_all[df_all["exclusion_criterion"] != "E1"].copy()
    return df_all


def _normalize_bed_columns_inplace(df):
    """Map legacy BED headers to canonical names used downstream."""
    if df is None or df.empty:
        return df
    low = {c.lower(): c for c in df.columns}

    def have(*opts):
        for o in opts:
            if o in low:
                return low[o]
        return None

    ren = {}
    if have('#chrom','chrom'): ren[have('#chrom','chrom')] = 'chrom'
    if have('start','start_diem_input'): ren[have('start','start_diem_input')] = 'start'
    if have('end','end_diem_input'):     ren[have('end','end_diem_input')]     = 'end'
    if have('qual'): ren[have('qual')] = 'qual'
    if have('ref'):  ren[have('ref')]  = 'ref'
    if have('seqalleles'): ren[have('seqalleles')] = 'SeqAlleles'
    if have('snv'):        ren[have('snv')]        = 'SNV'
    if have('nvnts','n(vnts)','n_vnts','nvariants'): ren[have('nvnts','n(vnts)','n_vnts','nvariants')] = 'nVNTs'
    if have('exclusion_criterion','exclusioncriterion'):
        ren[have('exclusion_criterion','exclusioncriterion')] = 'exclusion_criterion'
    if have('diem_genotype','dstring','dstrings'):
        ren[have('diem_genotype','dstring','dstrings')] = 'diem_genotype'

    df.rename(columns=ren, inplace=True)
    return df

    
# -------------------- Dstring & site helpers --------------------

def assert_dstring_shape(df: pd.DataFrame, n_samples: int, where: str="variants"):
    """Ensure diem_genotype is present and each row starts with 'S' and len==n+1."""
    if "diem_genotype" not in df.columns:
        raise ValueError(f"{where}: missing 'diem_genotype' column (check last column rename).")
    ok = df["diem_genotype"].str.startswith("S") & (df["diem_genotype"].str.len() == n_samples + 1)
    if not ok.any():
        # Give a short hint
        lens = df["diem_genotype"].str.len().value_counts().head(5).to_dict()
        raise ValueError(f"{where}: no valid Dstrings: startswith('S')={df['diem_genotype'].str.startswith('S').mean():.3f}, "
                         f"lengths(top)={lens}, expected={n_samples+1}")

def rows_by_chrom(df: pd.DataFrame, chroms: np.ndarray) -> Dict[str, pd.DataFrame]:
    out = {}
    for c in chroms:
        sub = df[df["chrom"].astype(str) == str(c)].copy()
        out[str(c)] = sub
    return out

# -------------------- Reconstruction --------------------

def OLDreconstruct_scaffold(ref_fa, scaffold_name, scaffold_len, rows, sample_names):
    """
    Build per-sample sequences for one scaffold using DIEM Dstrings and SeqAlleles.
    Policies:
      - Homozygous SNV: substitute base.
      - Heterozygous SNV: IUPAC code.
      - Homozygous indel (len(allele)!=len(ref)): apply exact replacement (length changes).
      - Heterozygous indel: conservative 'B' — DO NOT change length; leave ref segment
        intact and write a single 'N' at start position to flag ambiguity.
      - '_' (unencodable): skip / leave reference as-is at that site.
      - Overlapping edits: keep the first applied; skip later overlaps (counted).
    Returns: list[str] of reconstructed sequences (one per sample, same order as sample_names).
    """
    import numpy as np

    n = len(sample_names)

    # ---- DIEM decode (must match your distance code) -----------------------
    diemALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS'
    diemMaxVariants = 10
    diemMaxChars = diemMaxVariants * (diemMaxVariants + 1) // 2
    ABSij = (0, 0, 5, 5, 5, 9, 11, 13, 15, 17)
    MAXij = (0, 0, 0, 0, 3, 6, 10, 15, 21, 28)

    def enc(i, j):
        if max(i, j) >= diemMaxVariants: return None
        n_ = i + j + ABSij[abs(i - j)] + MAXij[max(i, j)]
        if 0 <= n_ < diemMaxChars:
            return diemALPHABET[n_]
        return None

    decode = {}
    for j in range(diemMaxVariants):
        for i in range(j + 1):
            ch = enc(i, j)
            if ch is not None:
                decode[ch] = (i, j)
    decode['_'] = None  # missing/unencodable

    # IUPAC for heterozygous SNVs
    IUPAC = {
        frozenset(("A","G")): "R",
        frozenset(("C","T")): "Y",
        frozenset(("G","C")): "S",
        frozenset(("A","T")): "W",
        frozenset(("G","T")): "K",
        frozenset(("A","C")): "M",
        frozenset(("A","C","G")): "V",
        frozenset(("A","C","T")): "H",
        frozenset(("A","G","T")): "D",
        frozenset(("C","G","T")): "B",
        frozenset(("A","C","G","T")): "N",
    }

    def iupac_for(a, b):
        a = a.upper(); b = b.upper()
        if a == b: return a
        return IUPAC.get(frozenset((a,b)), "N")

    # Fetch the reference once
    ref_seq = ref_fa.fetch(reference=scaffold_name, start=0, end=int(scaffold_len))
    # Per-sample builders as lists for efficiency
    builders = [list(ref_seq) for _ in range(n)]

    # Track which reference intervals we’ve already replaced (to avoid collisions)
    # We’ll mark applied intervals on a boolean mask; for long scaffolds this is memory-costly,
    # so we instead keep a list of applied intervals and do a tiny overlap check (counts are small).
    applied_intervals = []  # list of (start0, end0) in current coordinate space per sample?
    # NOTE: coordinates shift after indels. We therefore apply edits sample-by-sample.

    # Sort rows by BED start (0-based)
    rows = rows.sort_values("start", kind="mergesort")  # stable

    # Small counters for a one-line report in your notebook logs
    skipped_overlap = 0
    applied_hom_snv = applied_het_snv = 0
    applied_hom_indel = skipped_het_indel = 0

    # For each sample, apply edits in order
    for s_idx in range(n):
        seq = builders[s_idx]
        # We maintain a running offset that maps original BED coordinates to
        # current sequence coordinates as we change length via indels.
        offset = 0
        applied_intervals.clear()

        for _, r in rows.iterrows():
            ds = str(r.get("diem_genotype", ""))
            if not (ds and ds[0] == "S" and len(ds) == n + 1):
                continue  # malformed; ignore

            ch = ds[s_idx + 1]
            pair = decode.get(ch, None)
            if pair is None:
                continue  # '_' or invalid; leave reference

            start0 = int(r["start"]) + offset  # map to current coord space
            ref_allele = str(r.get("ref", "")) or "N"

            # Parse allele strings by DIEM rank (SeqAlleles is ordered by used ranks)
            seq_labels = str(r.get("SeqAlleles", "")).split(",")
            # Guard: if rank out of range, skip
            max_rank = len(seq_labels) - 1
            (i, j) = pair
            if i > max_rank or j > max_rank:
                continue  # cannot realize without labels

            a_i = seq_labels[i]
            a_j = seq_labels[j]

            # Determine homo/het & SNV/indel nature
            is_hom = (i == j)
            len_ref = len(ref_allele)
            len_i   = len(a_i) if a_i else 0
            len_j   = len(a_j) if a_j else 0

            # Compute the reference interval to replace (current coordinates)
            end0 = start0 + len_ref

            # Overlap check against already-applied intervals in this sample
            # (after coordinate shifts). We keep it simple: if the new [start0,end0)
            # intersects any prior [s,e), we skip this row for this sample.
            collided = False
            for (s,e) in applied_intervals:
                if not (end0 <= s or e <= start0):
                    collided = True; break
            if collided:
                skipped_overlap += 1
                continue

            # SNV if both ref and chosen allele length == 1
            def is_snv_len(x): return len(x) == 1

            if is_hom:
                # Homozygous call — realize exactly
                allele = a_i
                if is_snv_len(ref_allele) and is_snv_len(allele):
                    # SNV: simple substitution
                    seq[start0] = allele.upper() if allele else "N"
                    applied_hom_snv += 1
                    # interval length unchanged; record the 1bp site
                    applied_intervals.append((start0, start0+1))
                else:
                    # INDEL or multi-nt: replace ref segment with allele string
                    # 1) delete ref span
                    del seq[start0:end0]
                    # 2) insert allele
                    ins = list(allele.upper()) if allele else list("N")
                    for k, ch2 in enumerate(ins):
                        seq.insert(start0 + k, ch2)
                    # Update offset for downstream edits
                    delta = len(allele) - len_ref
                    offset += delta
                    applied_hom_indel += 1
                    # Record the *post-edit* occupied interval (use allele length)
                    applied_intervals.append((start0, start0 + len(allele)))
            else:
                # Heterozygous call
                if is_snv_len(ref_allele) and is_snv_len(a_i) and is_snv_len(a_j):
                    # Heterozygous SNV: write IUPAC at single base
                    seq[start0] = iupac_for(a_i, a_j)
                    applied_het_snv += 1
                    applied_intervals.append((start0, start0+1))
                else:
                    # Heterozygous involving indel → conservative 'B':
                    # do not change length; mark ambiguity at start with 'N'
                    seq[start0] = "N"
                    skipped_het_indel += 1
                    applied_intervals.append((start0, start0+1))
        # end for each row
        builders[s_idx] = "".join(seq)
    # end for each sample

    # Tiny summary to help you spot issues in the notebook
    print(
        f"[recon {scaffold_name}] hom_snv={applied_hom_snv}, het_snv={applied_het_snv}, "
        f"hom_indel={applied_hom_indel}, skipped_overlap={skipped_overlap}, "
        f"het_indel_conservative={skipped_het_indel}"
    )

    return builders

def reconstruct_scaffold(ref_fa, scaffold_name, scaffold_len, rows, sample_names):
    """
    Build per-sample sequences for one scaffold using DIEM Dstrings and SeqAlleles.

    Policies:
      - HOM SNV: substitute base.
      - HET SNV: IUPAC code.
      - HOM indel/multibase: exact replacement (length changes).
      - HET indel/multibase: conservative 'N' at start; do NOT change length.
      - Unencodable (including '_' and 'U' and any unknown site-state char): skip (leave ref).
      - Overlapping edits: keep first; skip later overlaps.
    """
    n = len(sample_names)

    decode, is_unencodable_char = build_diem_decoder()

    IUPAC = {
        frozenset(("A","G")): "R",
        frozenset(("C","T")): "Y",
        frozenset(("G","C")): "S",
        frozenset(("A","T")): "W",
        frozenset(("G","T")): "K",
        frozenset(("A","C")): "M",
        frozenset(("A","C","G")): "V",
        frozenset(("A","C","T")): "H",
        frozenset(("A","G","T")): "D",
        frozenset(("C","G","T")): "B",
        frozenset(("A","C","G","T")): "N",
    }

    def iupac_for(a, b):
        a = (a or "N").upper()
        b = (b or "N").upper()
        if a == b:
            return a
        return IUPAC.get(frozenset((a, b)), "N")

    ref_seq = ref_fa.fetch(reference=scaffold_name, start=0, end=int(scaffold_len))
    builders = [list(ref_seq) for _ in range(n)]

    rows = rows.sort_values("start", kind="mergesort")

    skipped_overlap = 0
    applied_hom_snv = applied_het_snv = 0
    applied_hom_indel = skipped_het_indel = 0

    for s_idx in range(n):
        seq = builders[s_idx]
        offset = 0
        applied_intervals = []

        for _, r in rows.iterrows():
            ds = str(r.get("diem_genotype", ""))
            if not (ds and ds[0] == "S" and len(ds) == n + 1):
                continue

            ch = ds[s_idx + 1]
            if is_unencodable_char(ch):
                continue

            pair = decode.get(ch, None)
            if pair is None:
                continue

            # Coordinates
            start0_ref = int(r["start"])  # 0-based BED start in reference coordinates
            start0 = start0_ref + offset  # current coordinate after indels applied
            ref_allele = str(r.get("ref", "")) or "N"
            len_ref = len(ref_allele)
            end0 = start0 + len_ref

            # Overlap check (current coordinate space)
            collided = False
            for (s, e) in applied_intervals:
                if not (end0 <= s or e <= start0):
                    collided = True
                    break
            if collided:
                skipped_overlap += 1
                continue

            seq_labels = str(r.get("SeqAlleles", "")).split(",")
            max_rank = len(seq_labels) - 1
            i, j = pair
            if i > max_rank or j > max_rank:
                continue

            a_i = seq_labels[i]
            a_j = seq_labels[j]
            is_hom = (i == j)

            def is_snv_len(x): 
                return isinstance(x, str) and len(x) == 1

            if is_hom:
                allele = a_i or "N"
                if is_snv_len(ref_allele) and is_snv_len(allele):
                    # HOM SNV
                    seq[start0] = allele.upper()
                    applied_hom_snv += 1
                    applied_intervals.append((start0, start0 + 1))
                else:
                    # HOM indel / multibase: replace ref segment with allele
                    del seq[start0:end0]
                    ins = list((allele or "N").upper())
                    for k, ch2 in enumerate(ins):
                        seq.insert(start0 + k, ch2)
                    delta = len(allele) - len_ref
                    offset += delta
                    applied_hom_indel += 1
                    applied_intervals.append((start0, start0 + len(allele)))
            else:
                # HET call
                if is_snv_len(ref_allele) and is_snv_len(a_i) and is_snv_len(a_j):
                    seq[start0] = iupac_for(a_i, a_j)
                    applied_het_snv += 1
                    applied_intervals.append((start0, start0 + 1))
                else:
                    # HET indel / multibase: conservative marker, no length change
                    seq[start0] = "N"
                    skipped_het_indel += 1
                    applied_intervals.append((start0, start0 + 1))

        builders[s_idx] = "".join(seq)

    print(
        f"[recon {scaffold_name}] hom_snv={applied_hom_snv}, het_snv={applied_het_snv}, "
        f"hom_indel={applied_hom_indel}, skipped_overlap={skipped_overlap}, "
        f"het_indel_conservative={skipped_het_indel}"
    )

    return builders


def reconstruct_region(ref_fa, chrom: str, region_start0: int, region_end0: int,
                       rows, sample_names):
    """
    Reconstruct only a reference slice [region_start0, region_end0) for each sample.

    Conservative rules at region boundaries:
      - only apply variants whose BED start is within region
      - skip edits that would extend beyond current region string after offsets
    """
    import pandas as pd

    # filter rows to those starting inside region
    rows = rows[(rows["start"] >= region_start0) & (rows["start"] < region_end0)].copy()
    if rows.empty:
        seq = ref_fa.fetch(reference=chrom, start=region_start0, end=region_end0)
        return [seq] * len(sample_names)

    # Convert to region-local coordinates so reconstruct_scaffold can work on a "mini-chrom"
    rows["start"] = rows["start"] - region_start0
    rows["end"]   = rows["end"]   - region_start0

    # Make a tiny fake scaffold: we pass a “length” equal to region length and fetch reference inside reconstruct_scaffold
    # To do that cleanly, temporarily monkeypatch fetch by wrapping ref_fa.
    class _SliceRef:
        def __init__(self, fa, chrom, start0, end0):
            self.fa = fa
            self.chrom = chrom
            self.start0 = start0
            self.end0 = end0
        def fetch(self, reference, start, end):
            # reference is ignored; start/end are relative to region
            return self.fa.fetch(reference=self.chrom, start=self.start0 + start, end=self.start0 + end)

    ref_slice = _SliceRef(ref_fa, chrom, region_start0, region_end0)
    region_len = region_end0 - region_start0

    # Use your existing reconstruct_scaffold (the “in line with DIEM decode” version you now have)
    return reconstruct_scaffold(ref_slice, chrom, region_len, rows, sample_names)


def rows_overlapping_regions(by_chr_all, regions):
    """
    regions: list[(chrom, start, end, label)]
    Returns a DataFrame of all rows overlapping any region.
    Assumes rows have 'start' (0-based) and represent a single-site call.
    """
    import pandas as pd

    parts = []
    # group regions by chrom for fewer filters
    reg_by_chr = {}
    for chrom, s, e, _lab in regions:
        reg_by_chr.setdefault(str(chrom), []).append((int(s), int(e)))

    for chrom, spans in reg_by_chr.items():
        df = by_chr_all.get(chrom)
        if df is None or df.empty:
            continue
        # rows are single-site; consider overlap by start within [s,e)
        # (if your rows are 1bp intervals, start is enough)
        s0 = df["start"].to_numpy()
        keep = None
        for s, e in spans:
            m = (s0 >= s) & (s0 < e)
            keep = m if keep is None else (keep | m)
        if keep is not None and keep.any():
            parts.append(df.loc[keep])

    if not parts:
        return pd.DataFrame(columns=["chrom","start","end","diem_genotype","SeqAlleles","ref"])
    return pd.concat(parts, ignore_index=True)


# -------------------- Heterozygosity summary --------------------

def het_summary_for_rows(rows: pd.DataFrame, sample_names: np.ndarray) -> Dict[str, dict]:
    """
    Count heterozygous SNVs / non-SNVs / unknown per sample for a scaffold.
    """
    n = len(sample_names)
    counts = {name: {"het_snv":0, "het_nonsnv":0, "Unencodable":0} for name in sample_names}
    for _, r in rows.iterrows():
        d = r["diem_genotype"]
        if not (isinstance(d, str) and d.startswith("S") and len(d) == n+1):
            continue
        seq_labels = (r.get("SeqAlleles","") or "").split(",")
        is_snv = len(seq_labels) >= 2 and len(seq_labels[0]) == 1 and len(seq_labels[1]) == 1
        for i, name in enumerate(sample_names):
            ch = d[1+i]
            if ch == "1":
                if is_snv:
                    counts[name]["het_snv"] += 1
                else:
                    counts[name]["het_nonsnv"] += 1
            elif ch == "_":
                counts[name]["Unencodable"] += 1
    return counts

# -------------------- Distances --------------------

def _pair_block_hamming(args):
    i0, i1, dstrings, n = args
    num = np.zeros((i1-i0, n), dtype=np.float64)
    den = np.zeros((i1-i0, n), dtype=np.int64)
    for d in dstrings:  # 'S'+n
        row = d[1:]
        for i in range(i0, i1):
            ci = row[i]
            if ci == "_": 
                continue
            for j in range(i, n):
                cj = row[j]
                if cj == "_":
                    continue
                num[i-i0, j] += (ci != cj)  # partial credit naturally by aggregating across sites
                den[i-i0, j] += 1
    return num, den

def _pair_block_compat(args):
    i0, i1, dstrings, n = args
    num = np.zeros((i1-i0, n), dtype=np.float64)
    den = np.zeros((i1-i0, n), dtype=np.int64)
    for d in dstrings:
        row = d[1:]
        for i in range(i0, i1):
            ci = row[i]
            for j in range(i, n):
                cj = row[j]
                # missing data allowed: if either is '_' skip (no evidence)
                if ci == "_" or cj == "_":
                    continue
                den[i-i0, j] += 1
                # compatible (distance 0) iff chars equal; otherwise 1
                num[i-i0, j] += (ci != cj)
    return num, den

def _reduce_symmetric(n: int, parts: List[Tuple[np.ndarray,np.ndarray]]) -> Tuple[np.ndarray,np.ndarray]:
    num = np.zeros((n,n), dtype=np.float64)
    den = np.zeros((n,n), dtype=np.int64)
    # parts order matches block index order; we reconstruct blocks by offsets
    # but we pass offsets alongside parts to be explicit:
    raise_on = False
    # Parts are returned in the same order they were mapped with offsets stored; we can reconstruct using zip
    # However simpler: we pass (num_block, den_block, i0, i1)
    # To keep above helpers simple, build alongside offsets in caller.
    return num, den  # placeholder (not used; we do reduction in the caller for clarity)


def OLDcompute_pairwise_from_dstrings(by_chr_dist, sample_names, mode="hamming", threads=1):
    """
    Multi-allelic Dstring distances.
    mode: "hamming" (unphased, partial credit) or "compat" (binary conflict).
    Returns (distance_matrix, coverage_matrix).
    """
    import numpy as np

    # ---- DIEM decode tables (must match vcf2diem) -------------------------
    diemALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS'
    diemMaxVariants = 10
    diemMaxChars = diemMaxVariants * (diemMaxVariants + 1) // 2
    ABSij = (0, 0, 5, 5, 5, 9, 11, 13, 15, 17)
    MAXij = (0, 0, 0, 0, 3, 6, 10, 15, 21, 28)

    # forward encoder (to build reverse map)
    def enc(i, j):
        if max(i, j) >= diemMaxVariants: return None
        n = i + j + ABSij[abs(i - j)] + MAXij[max(i, j)]
        if 0 <= n < diemMaxChars:
            return diemALPHABET[n]
        return None

    # reverse: char -> (i,j) with i<=j; '_' -> None
    decode = {}
    for j in range(diemMaxVariants):
        for i in range(j + 1):
            ch = enc(i, j)
            if ch is not None:
                decode[ch] = (i, j)
    decode['_'] = None  # missing

    # ---- scoring kernels ---------------------------------------------------
    def hamming_score(a, b):
        # a,b : tuples (i,j) or None
        if a is None or b is None:
            return None  # skip
        (i, j), (k, l) = a, b
        A = (i, j); B = (k, l)
        # same unordered set
        if (i == k and j == l) or (i == l and j == k):
            return 0.0
        # both homo different
        if i == j and k == l and i != k:
            return 1.0
        # one homo vs one het
        if i == j and k != l:
            return 0.5 if (i == k or i == l) else 1.0
        if i != j and k == l:
            return 0.5 if (k == i or k == j) else 1.0
        # both het: overlap?
        overlap = (i == k or i == l or j == k or j == l)
        return 0.5 if overlap else 1.0

    def compat_score(a, b):
        if a is None or b is None:
            return None
        (i, j), (k, l) = a, b
        # both homo different → conflict
        if i == j and k == l and i != k:
            return 1.0
        # otherwise conflict only if disjoint allele sets
        overlap = (i == k or i == l or j == k or j == l)
        return 0.0 if overlap else 1.0

    scorer = hamming_score if mode == "hamming" else compat_score

    # ---- accumulate --------------------------------------------------------
    n = len(sample_names)
    num   = np.zeros((n, n), dtype=np.float64)
    denom = np.zeros((n, n), dtype=np.float64)

    for chrom, rows in by_chr_dist.items():
        if rows is None or rows.empty:
            continue
        # only use well-formed Dstrings ('S' + n chars)
        D = rows['diem_genotype'].astype(str).to_numpy()
        D = [ds for ds in D if ds and ds[0] == 'S' and len(ds) == n + 1]
        if not D:
            continue

        # decode once: per-sample list of allele-pairs per site
        # pairs[sample_index][site_index] = (i,j) or None
        sites = len(D)
        pairs = [[None] * sites for _ in range(n)]
        for s_idx, ds in enumerate(D):
            for i in range(n):
                ch = ds[i + 1]
                pairs[i][s_idx] = decode.get(ch, None)

        # pairwise accumulate
        for i in range(n):
            Ai = pairs[i]
            for j in range(i, n):
                Aj = pairs[j]
                s = 0.0; c = 0.0
                for a, b in zip(Ai, Aj):
                    sc = scorer(a, b)
                    if sc is None:
                        continue
                    s += sc; c += 1.0
                num[i, j]   += s;   num[j, i]   += s
                denom[i, j] += c;   denom[j, i] += c

    with np.errstate(divide='ignore', invalid='ignore'):
        dist = num / denom
    # Diagonal: 0 when defined; NaN when no coverage
    for i in range(n):
        if denom[i, i] > 0:
            dist[i, i] = 0.0
    return dist, denom


def compute_pairwise_from_dstrings(by_chr_dist, sample_names, mode="hamming", threads=1):
    """
    Multi-allelic Dstring distances.
    mode:
      - "hamming": unphased, partial credit
      - "compat":  binary conflict (0 compatible, 1 incompatible), skipping Unencodable

    Returns:
      (distance_matrix, coverage_matrix) where distance is NaN if coverage==0.
    """
    import numpy as np

    decode, is_unencodable_char = build_diem_decoder()

    def hamming_score(a, b):
        if a is None or b is None:
            return None
        (i, j), (k, l) = a, b

        # Same unordered diplotype
        if (i == k and j == l) or (i == l and j == k):
            return 0.0

        # Both homozygous different
        if i == j and k == l and i != k:
            return 1.0

        # Homo vs het: partial if allele shared
        if i == j and k != l:
            return 0.5 if (i == k or i == l) else 1.0
        if i != j and k == l:
            return 0.5 if (k == i or k == j) else 1.0

        # Both het: partial if any allele shared
        overlap = (i == k or i == l or j == k or j == l)
        return 0.5 if overlap else 1.0

    def compat_score(a, b):
        if a is None or b is None:
            return None
        (i, j), (k, l) = a, b

        # Both homozygous different => definite conflict
        if i == j and k == l and i != k:
            return 1.0

        # Otherwise: compatible iff allele sets overlap
        overlap = (i == k or i == l or j == k or j == l)
        return 0.0 if overlap else 1.0

    scorer = hamming_score if mode == "hamming" else compat_score

    n = len(sample_names)
    num   = np.zeros((n, n), dtype=np.float64)
    denom = np.zeros((n, n), dtype=np.float64)

    for chrom, rows in by_chr_dist.items():
        if rows is None or rows.empty:
            continue

        D = rows["diem_genotype"].astype(str).to_numpy()
        D = [ds for ds in D if ds and ds[0] == "S" and len(ds) == n + 1]
        if not D:
            continue

        sites = len(D)

        # pairs[sample][site] = (i,j) or None
        pairs = [[None] * sites for _ in range(n)]
        for site_idx, ds in enumerate(D):
            # ds[1:] is per-sample site-state char
            for s_idx in range(n):
                ch = ds[s_idx + 1]
                if is_unencodable_char(ch):
                    pairs[s_idx][site_idx] = None
                else:
                    pairs[s_idx][site_idx] = decode.get(ch, None)

        # accumulate
        for i in range(n):
            Ai = pairs[i]
            for j in range(i, n):
                Aj = pairs[j]
                s = 0.0
                c = 0.0
                for a, b in zip(Ai, Aj):
                    sc = scorer(a, b)
                    if sc is None:
                        continue
                    s += sc
                    c += 1.0
                num[i, j]   += s
                denom[i, j] += c
                if i != j:
                    num[j, i]   += s
                    denom[j, i] += c

    with np.errstate(divide="ignore", invalid="ignore"):
        dist = num / denom

    # IMPORTANT: do NOT force diagonal to 0 if denom==0; keep NaN
    for i in range(n):
        if denom[i, i] > 0:
            dist[i, i] = 0.0

    return dist, denom



def run_summary(
    df_all: "pd.DataFrame",
    df_dist: "pd.DataFrame",
    chr_names,
    chr_lengths,
    sample_names,
    outdir: str,
    *,
    label: str = "diem2fasta",
    dstring_col: str = "diem_genotype",
    excl_col: str = "exclusion_criterion",
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    seqalleles_col: str = "SeqAlleles",
) -> dict:
    """
    Print a compact QC/diagnostic summary and return the same info as a dict.

    Assumptions:
      - df_all is your union (variants + excludes, with E1 removed if you do that)
      - df_dist is the subset actually used for distance matrices (e.g., E0-only)
      - Dstrings are site-strings 'S' + one char per sample
    """
    import os
    import numpy as np
    import pandas as pd

    n_samples = len(sample_names)
    expected_len = n_samples + 1

    def _is_good_dstring(x) -> bool:
        return isinstance(x, str) and len(x) == expected_len and x.startswith("S")

    # ---- Row counts
    n_all = 0 if df_all is None else int(len(df_all))
    n_dist = 0 if df_dist is None else int(len(df_dist))

    # ---- Exclusion counts (top few)
    excl_counts = None
    if df_all is not None and excl_col in df_all.columns:
        excl_counts = df_all[excl_col].astype("string").fillna("").value_counts().head(10)

    # ---- Dstring shape / quality
    good_all = bad_all = good_dist = bad_dist = 0
    if df_all is not None and dstring_col in df_all.columns:
        ds = df_all[dstring_col].tolist()
        good_all = sum(_is_good_dstring(x) for x in ds)
        bad_all  = len(ds) - good_all
    if df_dist is not None and dstring_col in df_dist.columns:
        ds = df_dist[dstring_col].tolist()
        good_dist = sum(_is_good_dstring(x) for x in ds)
        bad_dist  = len(ds) - good_dist

    # ---- Character composition on DIST rows (only well-formed)
    char_counts = None
    het1_total = unenc_total = 0
    if df_dist is not None and dstring_col in df_dist.columns and good_dist:
        # Concatenate all sample chars (drop leading 'S')
        # This is O(total_chars) but only for summary.
        concat = []
        for x in df_dist[dstring_col].tolist():
            if _is_good_dstring(x):
                concat.append(x[1:])
        all_chars = "".join(concat)
        if all_chars:
            from collections import Counter
            char_counts = Counter(all_chars)
            het1_total = char_counts.get("1", 0)
            unenc_total = char_counts.get("_", 0) + char_counts.get("U", 0)

    # ---- SNV vs non-SNV (best-effort)
    snv_rows = nonsnv_rows = unknown_rows = 0
    if df_dist is not None and seqalleles_col in df_dist.columns:
        sa = df_dist[seqalleles_col].astype("string")
        # SNV: first two alleles exist and are length 1
        # (best effort; empty/malformed -> unknown)
        for s in sa.tolist():
            if not s:
                unknown_rows += 1
                continue
            parts = str(s).split(",")
            if len(parts) >= 2 and len(parts[0]) == 1 and len(parts[1]) == 1:
                snv_rows += 1
            elif len(parts) >= 1:
                nonsnv_rows += 1
            else:
                unknown_rows += 1

    # ---- Chrom coverage
    chrom_row_counts = None
    chroms_present = 0
    if df_dist is not None and chrom_col in df_dist.columns:
        chrom_row_counts = (
            df_dist[chrom_col].astype("string").value_counts()
            if len(df_dist) else pd.Series(dtype="int64")
        )
        chroms_present = int((chrom_row_counts > 0).sum())

    # ---- Basic coordinate sanity
    coord_ok = None
    if df_dist is not None and start_col in df_dist.columns and end_col in df_dist.columns:
        try:
            starts = pd.to_numeric(df_dist[start_col], errors="coerce")
            ends   = pd.to_numeric(df_dist[end_col], errors="coerce")
            coord_ok = float(((starts.notna()) & (ends.notna()) & (ends >= starts)).mean())
        except Exception:
            coord_ok = None

    # ---- Build summary dict
    summary = {
        "label": label,
        "n_samples": n_samples,
        "expected_dstring_len": expected_len,
        "rows_all": n_all,
        "rows_dist": n_dist,
        "good_dstrings_all": good_all,
        "bad_dstrings_all": bad_all,
        "good_dstrings_dist": good_dist,
        "bad_dstrings_dist": bad_dist,
        "snv_rows_dist": snv_rows,
        "nonsnv_rows_dist": nonsnv_rows,
        "unknown_seqalleles_rows_dist": unknown_rows,
        "chroms_in_meta": int(len(chr_names)) if chr_names is not None else None,
        "chroms_with_dist_rows": chroms_present,
        "coord_sanity_fraction": coord_ok,
        "het_char_1_total": int(het1_total),
        "unenc_total_(_or_U)": int(unenc_total),
    }

    # ---- Print
    print(f"[{label}] samples: {n_samples} (expected Dstring len={expected_len})")
    print(f"[{label}] rows: all={n_all:,d}  dist={n_dist:,d}")
    print(f"[{label}] Dstrings: good/all={good_all:,d} bad/all={bad_all:,d} | good/dist={good_dist:,d} bad/dist={bad_dist:,d}")
    if coord_ok is not None:
        print(f"[{label}] coord sanity (end>=start & numeric): {coord_ok:.3f}")
    print(f"[{label}] dist rows by type: SNV={snv_rows:,d} non-SNV={nonsnv_rows:,d} unknown={unknown_rows:,d}")
    print(f"[{label}] chroms: meta={len(chr_names) if chr_names is not None else 'NA'}  with-dist-rows={chroms_present}")

    if excl_counts is not None and len(excl_counts):
        print(f"[{label}] exclusion_criterion (top 10):")
        for k, v in excl_counts.items():
            print(f"  {k or '<blank>'}\t{int(v):,d}")

    if char_counts is not None:
        # Show key chars first
        keys = ["0", "1", "2", "_", "U"]
        shown = {k: char_counts.get(k, 0) for k in keys}
        other = sum(v for k, v in char_counts.items() if k not in shown)
        print(f"[{label}] Dstring char totals (dist, excluding 'S'):")
        print("  " + "  ".join([f"{k}:{shown[k]:,d}" for k in keys]) + f"  other:{other:,d}")

    # ---- Write TSV (optional but nice)
    try:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, f"{label}_run_summary.tsv")
        pd.DataFrame([summary]).to_csv(out_path, sep="\t", index=False)
        summary["summary_tsv"] = out_path
    except Exception:
        summary["summary_tsv"] = None

    return summary

# -------------------- Orchestrator --------------------

def run_diem2fasta(meta_path: str,
                   variants_path: str,
                   ref_fasta_path: str,
                   excludes_path: Optional[str] = None,
                   outdir: str = "diem2fasta_out",
                   threads: int = 1,
                   out_fasta_path: Optional[str] = None,
                   regions_bed_path: Optional[str] = None
                   ) -> Dict[str, object]:
    """
    Notebook-friendly runner. Writes per-scaffold FASTAs and TSVs; returns dict.
    """
    os.makedirs(outdir, exist_ok=True)
    regions_fasta_written = None

    # META
    chr_names, chr_lengths, sample_names, _ploidy = load_meta(meta_path, ref_fasta_path, variants_path)
    n = len(sample_names)

    # BEDs (strings) and ensure Dstring column name
    df_v = read_bed_as_strings(variants_path)
    _normalize_bed_columns_inplace(df_v)
    df_x = None
    if excludes_path and os.path.exists(excludes_path):
        df_x = read_bed_as_strings(excludes_path)
        _normalize_bed_columns_inplace(df_x)

    # Normalise a few columns
    for df in (df_v, df_x) if df_x is not None else (df_v,):
        for c in ("start","end"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype("int64")
        if "exclusion_criterion" in df.columns:
            df["exclusion_criterion"] = df["exclusion_criterion"].astype("string").str.strip()

    # Union and keep everything except pure invariants (E1)
    df_all = union_variants(df_v, df_x)
    _normalize_bed_columns_inplace(df_all)

    # >>> INSERT #3: enforce required columns + numeric coords on the union <<<
    required = ['chrom', 'start', 'end', 'diem_genotype']
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        print("df_all columns:", list(df_all.columns))  # quick peek
        raise KeyError(f"Required columns missing after normalization: {missing}")
    df_all['chrom'] = df_all['chrom'].astype(str)
    for c in ('start', 'end'):
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce').astype('Int64').astype('int64')
    # <<< END INSERT #3 >>>
    
    # Assert Dstrings are well-formed (if any row exists)
    if not df_all.empty:
        assert_dstring_shape(df_all, n, where="variants+excludes union")

    # E0-only set for distances (matches your baseline)
    df_dist = df_all[df_all.get("exclusion_criterion","E0") == "E0"].copy()

    summary = run_summary(
        df_all=df_all,
        df_dist=df_dist,
        chr_names=chr_names,
        chr_lengths=chr_lengths,
        sample_names=sample_names,
        outdir=outdir,
        label="diem2fasta",
    )

    # Group by chromosome
    by_chr_all  = rows_by_chrom(df_all,  chr_names)
    by_chr_dist = rows_by_chrom(df_dist, chr_names)

    # Reference
    ref = pysam.FastaFile(ref_fasta_path)

    # --- Regions-only FASTA mode ------------------------------------------
    if regions_bed_path:
        print(f"[mode] regions2fasta: {regions_bed_path}")
        
        regions = read_regions_bed(regions_bed_path)
        if out_fasta_path is None:
            out_fasta_path = os.path.join(outdir, "regions.recon.fasta")
        regions_fasta_written = out_fasta_path

        with open(out_fasta_path, "w", encoding="utf-8") as fh:
            for chrom, rstart, rend, label in regions:
                rows = by_chr_all.get(str(chrom))
                if rows is None or rows.empty:
                    seq = ref.fetch(reference=str(chrom), start=int(rstart), end=int(rend))
                    for sname in sample_names:
                        fh.write(f">{label}|{sname}\n")
                        for i in range(0, len(seq), 60):
                            fh.write(seq[i:i+60] + "\n")
                    continue

                seqs = reconstruct_region(ref, str(chrom), int(rstart), int(rend), rows, sample_names)
                for sname, seq in zip(sample_names, seqs):
                    fh.write(f">{label}|{sname}\n")
                    for i in range(0, len(seq), 60):
                        fh.write(seq[i:i+60] + "\n")

        # Still compute het report + distances as usual (or skip, if you want)
        # Return now so NO per-scaffold FASTAs are written.
        het_df = pd.DataFrame({
            "sample": sample_names,
            "het_snv": 0,
            "het_nonsnv": 0,
            "Unencodable": 0,
        })
        # (Optional) keep your existing het computation if you want it in regions-only mode.

        d_ham, cov_ham   = compute_pairwise_from_dstrings(by_chr_dist, sample_names, mode="hamming", threads=threads)
        d_comp, cov_comp = compute_pairwise_from_dstrings(by_chr_dist, sample_names, mode="compat",  threads=threads)

        pd.DataFrame(d_ham,  index=sample_names, columns=sample_names).to_csv(
            os.path.join(outdir, "distance_unphased_hamming.tsv"), sep="\t", na_rep=""
        )
        pd.DataFrame(cov_ham, index=sample_names, columns=sample_names).to_csv(
            os.path.join(outdir, "distance_unphased_hamming_coverage.tsv"), sep="\t"
        )
        pd.DataFrame(d_comp, index=sample_names, columns=sample_names).to_csv(
            os.path.join(outdir, "distance_compatibility.tsv"), sep="\t", na_rep=""
        )
        pd.DataFrame(cov_comp, index=sample_names, columns=sample_names).to_csv(
            os.path.join(outdir, "distance_compatibility_coverage.tsv"), sep="\t"
        )

        print("[note] regions2fasta mode: wrote ONLY regions FASTA (no per-scaffold FASTAs).")

        # Het report in regions-only mode
        from collections import Counter
        het_totals = {name: Counter() for name in sample_names}
        
        df_reg = rows_overlapping_regions(by_chr_all, regions)
        if not df_reg.empty:
            hs = het_summary_for_rows(df_reg, sample_names)
            for name in sample_names:
                het_totals[name].update(hs[name])
        
        het_df = pd.DataFrame({
            "sample": sample_names,
            "het_snv":     [het_totals[nm].get("het_snv",0) for nm in sample_names],
            "het_nonsnv":  [het_totals[nm].get("het_nonsnv",0) for nm in sample_names],
            "Unencodable": [het_totals[nm].get("Unencodable",0) for nm in sample_names],
        })
        het_df.to_csv(os.path.join(outdir, "regions_heterozygosity_report.tsv"), sep="\t", index=False)

        return {
            "regions_fasta": out_fasta_path,
            "het_report": het_df,
            "distance_hamming": pd.DataFrame(d_ham, index=sample_names, columns=sample_names),
            "coverage_hamming": pd.DataFrame(cov_ham, index=sample_names, columns=sample_names),
            "distance_compat": pd.DataFrame(d_comp, index=sample_names, columns=sample_names),
            "coverage_compat": pd.DataFrame(cov_comp, index=sample_names, columns=sample_names),
        }
    # ----------------------------------------------------------------------


    # FASTAs + heterozygosity
    from collections import Counter
    het_totals = {name: Counter() for name in sample_names}
    fasta_paths = []

    for chrom, clen in zip(chr_names, chr_lengths):
        rows = by_chr_all.get(str(chrom))
        if rows is None or rows.empty:
            # write reference for all samples
            seq = ref.fetch(reference=str(chrom), start=0, end=int(clen))
            outfa = os.path.join(outdir, f"{chrom}.recon.fasta")
            with open(outfa, "w") as fh:
                for sname in sample_names:
                    fh.write(f">{sname}\n")
                    for i in range(0, len(seq), 60):
                        fh.write(seq[i:i+60] + "\n")
            fasta_paths.append(outfa)
            continue

        seqs = reconstruct_scaffold(ref, str(chrom), int(clen), rows, sample_names)
        outfa = os.path.join(outdir, f"{chrom}.recon.fasta")
        with open(outfa, "w") as fh:
            for sname, seq in zip(sample_names, seqs):
                fh.write(f">{sname}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i:i+60] + "\n")
        fasta_paths.append(outfa)

        hs = het_summary_for_rows(rows, sample_names)
        for name in sample_names:
            het_totals[name].update(hs[name])

    het_df = pd.DataFrame({
        "sample": sample_names,
        "het_snv":     [het_totals[nm].get("het_snv",0) for nm in sample_names],
        "het_nonsnv":  [het_totals[nm].get("het_nonsnv",0) for nm in sample_names],
        "Unencodable": [het_totals[nm].get("Unencodable",0) for nm in sample_names],
    })
    het_df.to_csv(os.path.join(outdir, "heterozygosity_report.tsv"), sep="\t", index=False)

    # DEBUG FUCK
    # --- Dstring sanity diagnostics (E0 rows only) ---
    def _is_d_ok(s, n):
        # proper Dstring: a string, starts with 'S', length == n+1
        return isinstance(s, str) and s.startswith('S') and len(s) == (n + 1)

    n_e0 = sum((not df.empty) for df in by_chr_dist.values())
    ok = 0
    bad = 0
    for chrom, df in by_chr_dist.items():
        if df is None or df.empty:
            continue
        col = 'diem_genotype' if 'diem_genotype' in df.columns else None
        if col is None:
            continue
        s = df[col].astype(object)
        mask_ok = s.map(lambda x: _is_d_ok(x, n))
        ok += int(mask_ok.sum())
        bad += int((~mask_ok).sum())

    print(f"[diag] E0 contigs with rows: {n_e0}; good Dstrings: {ok}; bad Dstrings: {bad}")
    if ok == 0:
        print("[diag] No usable site Dstrings were found in the E0 set. "
              "Expect NaNs in distance matrices and zero coverages.")

    #END DEBUG FUCK
    
    # Distances from Dstrings (E0 only) — your two flavors
    d_ham, cov_ham   = compute_pairwise_from_dstrings(by_chr_dist, sample_names, mode="hamming", threads=threads)
    d_comp, cov_comp = compute_pairwise_from_dstrings(by_chr_dist, sample_names, mode="compat",  threads=threads)

    pd.DataFrame(d_ham,  index=sample_names, columns=sample_names).to_csv(
        os.path.join(outdir, "distance_unphased_hamming.tsv"), sep="\t", na_rep=""
    )
    pd.DataFrame(cov_ham, index=sample_names, columns=sample_names).to_csv(
        os.path.join(outdir, "distance_unphased_hamming_coverage.tsv"), sep="\t"
    )
    pd.DataFrame(d_comp, index=sample_names, columns=sample_names).to_csv(
        os.path.join(outdir, "distance_compatibility.tsv"), sep="\t", na_rep=""
    )
    pd.DataFrame(cov_comp, index=sample_names, columns=sample_names).to_csv(
        os.path.join(outdir, "distance_compatibility_coverage.tsv"), sep="\t"
    )

    # Policy note
    print("[note] Heterozygous non-SNV sites were written as 'N' (conservative).")

    return {
        "fasta_paths": fasta_paths,
        "het_report": het_df,
        "distance_hamming": pd.DataFrame(d_ham, index=sample_names, columns=sample_names),
        "coverage_hamming": pd.DataFrame(cov_ham, index=sample_names, columns=sample_names),
        "distance_compat": pd.DataFrame(d_comp, index=sample_names, columns=sample_names),
        "coverage_compat": pd.DataFrame(cov_comp, index=sample_names, columns=sample_names),
        "regions_fasta": regions_fasta_written,
    }

# -------------------- CLI (optional) --------------------
if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        raise SystemExit("diem2fasta: refusing to run CLI inside ipykernel; call run_diem2fasta(...) instead.")
    import argparse
    ap = argparse.ArgumentParser(description="diem2fasta (clean)")
    ap.add_argument("--meta",     required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--ref",      required=True)
    ap.add_argument("--excludes", default=None)
    ap.add_argument("--outdir",   default="diem2fasta_out")
    ap.add_argument("--threads",  type=int, default=max(1, cpu_count()//2))
    ap.add_argument("--out_fasta", default=None, help="Write a single combined FASTA to this path.")
    ap.add_argument("--regions2fasta", default=None, help="BED of regions to reconstruct to FASTA (0-based). Col4 becomes label.")

    args = ap.parse_args()
    run_diem2fasta(
        meta_path=args.meta,
        variants_path=args.variants,
        ref_fasta_path=args.ref,
        excludes_path=args.excludes,
        outdir=args.outdir,
        threads=args.threads,
        out_fasta_path=args.out_fasta,
        regions_bed_path=args.regions2fasta,
    )
