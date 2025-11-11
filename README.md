# diemPy

A Python package for genome polarization and subsequent analyses using the diem (Diagnostic Index for the Expectation Maximization) method.

## Overview

diemPy is a computational tool designed to polarize genomic data for hybrid zone analysis. The package implements an expectation-maximization (EM) algorithm to determine the optimal polarization of genetic markers, enabling researchers to identify and analyze patterns of introgression and hybridization in genomic datasets.

### Key Features
- **VCF Processing**: Direct conversion from VCF files to diem format
- **Genome Polarization**: Automated polarization of genetic markers using EM algorithm
- **Thresholding**: Tools to help decide on threshold for minium diagnostic index of markers to retain
- **Kernel Smoothing**: Spatial smoothing of genomic data along chromosomes
- **Obtaining Tract Length**: Detection and analysis of genomic regions with consistent ancestry
- **Parallel Processing**: Multi-core support for computationally intensive operations
- **Flexible I/O**: Support for various input/output formats including BED-like files

### Core Functionality

The package provides several main analysis workflows:

1. **Data Import and Processing**
   - Convert VCF files to diem format using [`vcf2diem`](src/diem/vcf2diembeds.py)
   - Read diem BED format files with [`read_diem_bed`](src/diem/io.py)
   - Handle masking of individuals and sites
   - Handling correct ploidy information, e.g. with regard to sex chromosomes

2. **Polarization Analysis**
   - Initialize polarization using random null
   - Run EM algorithm to optimize marker polarization
   - Calculate diagnostic indices and support values
   - Parallel and linear processing options available

3. **Post-Polarization Analysis**
   - Compute hybrid indices for individuals
   - Apply thresholding to filter less informative markers
   - Perform kernel smoothing across genomic windows
   - Generate tracts of contiguous ancestry and store them as contigs

## Core Classes and Functions

### DiemType Class
The central data structure that holds:
- **DMBC**: Diem Matrix By Chromosome (state matrices)
- **Genomic positions** and chromosome information
- **Individual metadata** including ploidy and exclusions
- **Polarization results** including polarity, diagnostic indices, and support values
- **Contigs** which are per-chromosome per-interval lists of tracts 

Key methods:
- [`polarize()`](src/diem/diemtype.py): Run polarization analysis
- [`apply_threshold()`](src/diem/diemtype.py): Filter markers by diagnostic index
- [`smooth()`](src/diem/diemtype.py): Apply kernel smoothing
- [`sort()`](src/diem/diemtype.py): Sort individuals by hybrid index
- ['create_contig_matrix()](src/diem/diemtype.py): builds the contigs from the processed data



## Installation and Getting Started


To install `diem`, first set up a new conda environment:

```sh
conda create -n diem python=3.11
conda activate diem
```

install `diem` using pip. From within your conda environment, and in the diem directory (where the pyproect.toml file is located) and run :


```sh
python -m pip install .
```

`diem` can be installed as a package and used in any python script. This means that the big tasks like polarization can be offloaded to a computing cluster. However, for many datasets, `diem` can run on a modern laptop. We recommend using `diem` and exploring the data within a jupyter notebook, and we provide an example workflow as a starting point (notebooks/derek_input_and_process_output.ipynb).  To install install and run jupyter lab:

```sh
conda install -c conda-forge jupyterlab
```  

Copy the example notebook (notebooks/derek_input_and_process_output.ipynb) and paste it into a directory where you will store and analyse your data. Then, in a terminal, navigate to your root directory and run jupyter lab. This will open a new browser tab or window.

```sh
cd ~
jupyter lab
```

From within jupyter lab, on the left, navigate to your working directory and open the example notebook.  



## Basic Usage

We recommend you check out the example notebook.  However, here is a minium set of commands to perform a full processing of the data

```python
import diem
from diem.io import read_diem_bed, write_polarized_bed 
from diem.contigs import export_contigs_to_ind_bed_files
from diem.diemtype import save_DiemType, load_DiemType

# Load data
d = read_diem_bed('input.bed', 'meta.bed')

# Polarize the data
d_polarized = d.polarize(ncores=4, maxItt=500)

# output polarized data
write_polarized_bed('input.bed','output.bed',d_polarized)

# Apply threshold filtering
d_filtered = d_polarized.apply_threshold(threshold=0.5)

# Sort the filtered data by hybrid index
d_filtered.sort()

# Smooth the data
d_smooth = d_filtered.smooth(scale=0.1)

# Create contig matrix
d_smooth.create_contig_matrix()

# Output contigs as bed files
export_contigs_to_ind_bed_files('outputdir/')
```

Note that at any time, a diemtype object may be saved or loaded directly. This is fast and uses less memory, but it is only readable from within the diem package.

```python
#save the smoothed diemtype object
save_DiemType('output.diemtype',d_smooth)

#load the diemtype back in 
d_smooth = load_DiemType('output.diemtype')
```

## Command Line Tools

The package includes a command-line tool for VCF conversion:

```bash
vcf2diem input.vcf
```

This generates three output files:
- `input.vcf.diem_input.bed`: Variant input for diem analysis
- `input.vcf.diem_exclude.bed`: Sites excluded from analysis
- `input.vcf.diem_meta.bed`: Metadata for diem analysis

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- numba >= 0.56.0
- matplotlib >= 3.4.0
- docopt >= 0.6.2
- scikit-allel >= 1.3.0

## Documentation

Additional documentation and examples can be found in the [`docs/`](docs/) directory, including:
- [Workflow development notebook](docs/workflow_dev_test.ipynb)
- [API reference](docs/api.md)
- [Quickstart guide](docs/quickstart.md)

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Author

Derek Setter (derek.setter@gmail.com)

## Citation

*[Add citation information when available]*