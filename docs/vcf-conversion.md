# VCF Conversion

The package includes a command-line tool for VCF conversion. input.vcf is the path to your vcf file.

```bash
vcf2diem input.vcf
```

This generates three output files placed in the same directory as the input vcf:
- `input.vcf.diem_input.bed`: Variant input for diem analysis
- `input.vcf.diem_exclude.bed`: Sites excluded from analysis
- `input.vcf.diem_meta.bed`: Metadata for diem analysis

# Updating the Ploidy

vcf2diem automatically generates a meta data file. Each row is a chromosome. The first column are the chromosome names. The second column contains chromosome lengths in base pairs obtained from the vcf file.  

The remaining columns correspond to a single individual.  Each entry describes the ploidy of the individual for that chromosome. By default, all chromosomes are assumed to be diploid. However, this may not be the case.  For example, in XY systems, the X chromosome is haploid in males.  

To correct this the user may provide a file with the first column header #Ind, and list each individual in the dataset underneath.

Each subsequent column should have a chromosome name at the top, with the corresponding ploidy for each individual below. For example:

```text
#Ind    chrName1  chrName2
i1      1          1
i2      2          2
i3      2          2
```

There is a built-in function of diempy to run at the start of the workflow (see workflow example notebook for details) that will update the meta data file. This should be done before the polarization and subsequent analysis. 