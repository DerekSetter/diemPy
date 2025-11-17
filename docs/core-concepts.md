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

There are three main classes in Diem.  Namely, 
- **DiemType** which is the main data structure that holes the core functions for polarizing the data
- **Contig** class which simply holds the sequence of intervals that form a contig
- **Intervals** a class which describes a single interval, i.e. a contiguous region of a single state in a given individual 

**DiemType** → **Contig** → **Interval**: DiemType contains the diem-formatted data as well as a matrix of Contigs (one per individual per chromosome), each Contig contains a list of Intervals representing contiguous ancestry tracts.

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

### Contig Class
The [`Contig`](src/diem/contigs.py) class represents a collection of genomic intervals for a specific individual and chromosome, essentially describing the complete ancestry structure along that chromosome.

Key attributes:
- **chrName**: Chromosome name
- **indName**: Individual name
- **intervals**: List of Interval objects making up the contig
- **num_intervals**: Number of intervals in the contig

Key methods:
- [`printIntervals()`](src/diem/contigs.py): Display interval information in a readable format
- [`get_my_intervals_of_state()`](src/diem/contigs.py): Filter intervals by ancestry state

### Interval Class
The [`Interval`](src/diem/contigs.py) class represents a contiguous genomic region with consistent ancestry state for a specific individual and chromosome.

Key attributes:
- **chrName**: Chromosome name
- **indName**: Individual name  
- **idxl, idxr**: Left and right indices (inclusive) in the state matrix
- **l, r**: Left and right physical positions
- **state**: Ancestry state of the interval (0=uncalled, 1-3=called states)

Key methods:
- [`span()`](src/diem/contigs.py): Calculate the physical span of the interval
- [`mapSpan()`](src/diem/contigs.py): Calculate the relative span as fraction of chromosome length
