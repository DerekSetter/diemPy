# Advanced Polarizing Options

Here, we highlight three advanced options for polarizing that the user can specify using additional input files

1. specifying initial polarity
2. masking individuals
3. masking sites



## Specifying the initial polarity

It may at times be useful to start the EM algorithm with a pre-defined random null. To do this, we simply store the initial null in our *raw* diemtype object before polarizing.  `diem` will automatically use this as the initial state.

We must specify the polarity for each position in the genome. We do this in a tab-separated txt file specifying the chromosome names, positions, and polarities, e.g.:

```text
chromosome  position    polarity
contig_83   118     1
contig_83   131     0
contig_83   151     0
contig_84   183     1
```

Note that the chromosomes/contigs may be in any order, however, for each the positions must be sorted and match those within the raw diemtype object. 

```python
dRaw = diem.read_diem_bed('input.bed','meta.bed')
dRaw.add_initial_polarity('initialPolarity.txt')
dPol = dRaw.polarize()
```

## Masking Individuals and Sites

We can apply masks to individuals and sites during polarization.  Masked sites and Masked individuals do not influence polarization, but they do themselves get polarized. In terms of the `diem` algorithm, this means that masked individuals and masked sites are excluded when calculating the I4 matrix that describes the barrier.  

### individual masks:

Individuals to mask can be specified in a text file containing a single column of the names of individuals to exclude:

```text
individual_name_1
individual_name_2
```

We update the *raw* diemtype with this information, and those individuals will automatically be masked by `diem`:

```python
dRaw = diem.read_diem_bed('input.bed','meta.bed')
dRaw.add_individual_exclusions('inidividualsMasked.txt')
dPol = dRaw.polarize()
```

The bed format output of diem includes a header line that indicates which individuals were masked during the polarization step. 

### sites masking

Sites to mask can be specified in a bed format file, each line indicating a chromosome (by name) and the region to mask (0-indexed). A single site can be masked by specifying a one-base-pair region. In this example we mask a large segment of chromosome 2 and two specific sites in chromosome 3:

```text
chromosome_2    0   341001
chromosome_3    281 282
chromosome_3    12999   13000
```

We apply this mask to the raw diemtype object before polarizing and it will automatically be included in the EM step.

```python
dRaw = diem.read_diem_bed('input.bed','meta.bed')
dRaw.add_site_exclusions('sitesMasked.txt')
dPol = dRaw.polarize()
```

## Combining all three

Note that it is possible to mix-and-match these advanced options, E.g. to apply all three at once during polarizing, simpy use:

```python
dRaw = diem.read_diem_bed('input.bed','meta.bed')
dRaw.add_initial_polarity('initialPolarity.txt')
dRaw.add_individual_exclusions('inidividualsMasked.txt')
dRaw.add_site_exclusions('sitesMasked.txt')
dPol = dRaw.polarize()
```