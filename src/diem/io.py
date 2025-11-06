import pandas as pd
import numpy as np
from . import diemtype as dt

# this is a sketch of a function for importing the bed file data and constructing a diemtype object from it
# uses the output of vcf2diembed version from Mid october that stuart sent us  

# AI generated faster version of read_diem_bed approximately 2x speed up
def read_diem_bed_fast(bed_file_path, meta_file_path):
    """
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
    
    # Create chromosome lookup for faster filtering
    chr_groups = df_bed.groupby('chrom')
    
    # Fixed vectorized genotype mapping
    def map_gt_vectorized(gt_strings):
        """Vectorized genotype mapping - handles '_' and other characters properly"""
        # Convert to numpy array of characters, skipping first character
        gt_chars = np.array([list(s)[1:] for s in gt_strings])
        
        # Create result array
        result = np.zeros(gt_chars.shape, dtype=np.int8)
        
        # Map characters using boolean indexing (much safer than ASCII lookup)
        result[gt_chars == '0'] = 1
        result[gt_chars == '1'] = 2
        result[gt_chars == '2'] = 3
        # Everything else (including '_') stays as 0
        
        return result.T
    
    # Process data by chromosome - avoid repeated filtering
    positionByChr = []
    DMBC = []
    
    if hasPolarity:
        polarityByChr = []
        initialPolByChr = []
        DIByChr = []
        supportByChr = []
        siteExclusionsByChr = []
        allExclusionsAreNone = True
    
    for chr in chrNames:
        thisDF = chr_groups.get_group(chr)
        
        # Positions
        positions = thisDF['end'].values.astype(int)
        positionByChr.append(positions)
        
        # Genotypes - vectorized processing
        allele_matrix = map_gt_vectorized(thisDF['diem_genotype'].values)
        DMBC.append(allele_matrix)
        
        if hasPolarity:
            # Process all polarity-related data at once
            polarityByChr.append(thisDF['polarity'].values.astype(int))
            initialPolByChr.append(thisDF['nullPolarity'].values.astype(int))
            DIByChr.append(thisDF['DI'].values.astype(np.float64))
            supportByChr.append(thisDF['Support'].values.astype(np.float64))
            
            # Site exclusions
            masked_vals = thisDF['masked'].values.astype(int)
            if np.all(masked_vals == 0):
                siteExclusionsByChr.append(None)
            else:
                siteExclusionsByChr.append(np.where(masked_vals == 1)[0])
                allExclusionsAreNone = False
    
    # Create DiemType object
    d = dt.DiemType(DMBC, sampleNames, ploidyByChr, chrNames, positionByChr, chrLengths)
    d.indExclusions = individualsMasked
    
    if hasPolarity:
        d.PolByChr = polarityByChr
        d.DIByChr = DIByChr
        d.SupportByChr = supportByChr
        d.initialPolByChr = initialPolByChr
        
        if allExclusionsAreNone:
            d.siteExclusionsByChr = None
        else:
            d.siteExclusionsByChr = siteExclusionsByChr
        
        # Vectorized polarity flipping
        for idx in range(len(chrNames)):
            d.DMBC[idx] = dt.flip_polarity(d.DMBC[idx], d.PolByChr[idx])
    
    return d


def read_diem_bed(bed_file_path,meta_file_path):
    """
    Reads a diem BED file and returns a DiemType object. If the bed file has been processed by diem,
    the bed file will have a preamble indicating any individuals that were masked during polarization, and it will contain columns for initial Polarity, end polarity, diagnostic index, support, and a column indicating whether the site was masked or not.
    if those fields exist, those attributes will be added to the DiemType object and the diemmatrix will be flipped to match that polarity.
    
    Parameters:
    bed_file_path (str): Path to the diem BED file.
    meta_file_path (str): Path to the diem metadata file.

    Returns:
    DiemType: DiemType object containing the diem BED data.

    """


    df_meta = pd.read_csv(meta_file_path, sep='\t')
    chrNames = np.array(df_meta['#Chrom'].values)
    
    #print(chrNames)
    ### For lengths, this is wrong!  just a placeholder. need to chnage to length when this is available ###
    chrLengths = np.array(df_meta['RefEnd0'].values) - np.array(df_meta['RefStart0'].values)

    #print(chrLengths)
    sampleNames = np.array(df_meta.columns[6:])
    #print(sampleNames)
    #Assuming first three columns are #Chrom, Start_diem_input, End_diem_input, n(diem_inputs)

    ploidyByChr = []
    for chr in chrNames:
        row = df_meta[df_meta['#Chrom'] == chr]
        ploidy = np.array(row.iloc[0,6:].values, dtype=int)
        ploidyByChr.append(ploidy)
    #print(ploidyByChr)


    #the input bed file:

    #read preamble line for individual exclusions. This line also indicates whether polarity information is present or not.

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
                        individualsMasked=clean_line.split(',')
                nSkipLines += 1
            else:
                break
    
    if len(preamble)>0:
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

    
    df_bed = pd.read_csv(bed_file_path, sep='\t', names=column_names,skiprows=nSkipLines+1) 

    positionByChr = []
    for chr in chrNames:
        thisDF = df_bed[df_bed['chrom'] == chr]
        positions = thisDF['end'].values.tolist()
        positions = np.array(positions,dtype=int)
        positionByChr.append(positions)

    def map_gt_to_diem(gt_char):
        if gt_char == '0':
            return 1
        elif gt_char == '1':
            return 2
        elif gt_char == '2':
            return 3
        else:
            return 0
        
    DMBC = []
    for chr in chrNames:
        thisDF = df_bed[df_bed['chrom'] == chr]
        allele_matrix = [list(s)[1:] for s in thisDF['diem_genotype']]
        allele_matrix = np.array([[map_gt_to_diem(gt) for gt in s] for s in allele_matrix],dtype = np.int8)
        allele_matrix = allele_matrix.transpose()  # transpose to shape (n_samples, n_variants)
        DMBC.append(allele_matrix)

    #print(DMBC[0][:,:10])  # print first 10 variants for first chromosome
    # construct the DiemType object, then add any additional information from polarization if it exists.
    d = dt.DiemType(DMBC,sampleNames,ploidyByChr,chrNames,positionByChr,chrLengths)

    d.indExclusions = individualsMasked
    if hasPolarity:
        polarityByChr = []
        initialPolByChr = []
        DIByChr = []
        supportByChr = []
        siteExclusionsByChr = []

        allExclusionsAreNone = True
        for chr in chrNames:
   
            thisDF = df_bed[df_bed['chrom'] == chr]
            
            polarity = thisDF['polarity'].values.tolist()
            polarity = np.array(polarity,dtype=int)
            polarityByChr.append(polarity)

            initialPol = thisDF['nullPolarity'].values.tolist()
            initialPol = np.array(initialPol,dtype=int)
            initialPolByChr.append(initialPol)

            DI = thisDF['DI'].values.tolist()
            DI = np.array(DI,dtype=np.float64)
            DIByChr.append(DI)

            support = thisDF['Support'].values.tolist()
            support = np.array(support,dtype=np.float64)
            supportByChr.append(support)

            siteExclusions_array = thisDF['masked'].values.tolist()
            siteExclusions_array = np.array(siteExclusions_array,dtype=int)
            if np.all(siteExclusions_array == 0):
                siteExclusionsByChr.append(None)
            else:
                siteExclusions_array = np.where(siteExclusions_array == 1)[0]
                #siteExclusions_array = siteExclusions_array[siteExclusions_array != 0]
                siteExclusionsByChr.append(siteExclusions_array)
                allExclusionsAreNone = False


        if allExclusionsAreNone:
            d.siteExclusionsByChr = None
        else:
            d.siteExclusionsByChr = siteExclusionsByChr

        d.PolByChr = polarityByChr
        d.DIByChr = DIByChr
        d.SupportByChr = supportByChr
        d.initialPolByChr = initialPolByChr

        # Now flip the DMBC according to polarity
        for idx in range(len(chrNames)):
            d.DMBC[idx] = dt.flip_polarity(d.DMBC[idx],d.PolByChr[idx])

    return d

# note: by copying the input bed file and adding columns to it, we preserve all original information. It also means that even if diemtype object being saved has had its individuals reordered by HI, the output bed file will still have the original order.  
# if we store all the added info in the diemtype object so that we do not need to re-read the input bed file when saving the polarized data, we also need to keep track of whether the data has been reordered or not.
def write_polarized_bed(inputFilePath, outputFilePath, diemTypeObj):
    """
    Writes a polarized DiEM BED file based on the provided DiemType object.
    
    Parameters:
    inputFilePath (str): Path to the original DiEM BED file.
    outputFilePath (str): Path to the output polarized DiEM BED file.
    diemTypeObj (DiemType): DiemType object containing polarity information.
    """
    

    if diemTypeObj.PolByChr is None:
        raise ValueError("DiemType object does not contain polarity information.")
    if inputFilePath is None or outputFilePath is None:
        raise ValueError("Input and output file paths must be provided.")
    if inputFilePath == outputFilePath:
        raise ValueError("Input and output file paths must be different to avoid overwriting.")

    # produce a column for sites excluded
    sitesMasked = [np.array([0]*len(diemTypeObj.posByChr[i]),dtype=int) for i in range(len(diemTypeObj.chrNames))]

    if diemTypeObj.siteExclusionsByChr is None:
        pass
    else:
        for i in range(len(diemTypeObj.chrNames)):
            if diemTypeObj.siteExclusionsByChr[i] is not None:
                sitesMasked[i][diemTypeObj.siteExclusionsByChr[i]] = 1
    
    preambleLines = []
    if diemTypeObj.indExclusions is not None:
        masked_inds = '##IndividualsMasked='+','.join(diemTypeObj.indExclusions)
        preambleLines.append(masked_inds+"\n")
    else:
        preambleLines.append('##IndividualsMasked=None\n')

    #I need to write those preamble lines, then place the dataframe information below the preamble
    with open(outputFilePath, 'w') as f_out:
        for line in preambleLines:
            f_out.write(line)

    #how do I append to an existing file with pandas? 
    df_bed = pd.read_csv(inputFilePath, sep='\t') 
    df_bed['NullPolarity'] = np.hstack(diemTypeObj.initialPolByChr)
    df_bed['Polarity'] = np.hstack(diemTypeObj.PolByChr)
    df_bed['DiagnosticIndex'] = np.hstack(diemTypeObj.DIByChr)
    df_bed['Support'] = np.hstack(diemTypeObj.SupportByChr)
    df_bed['SiteMasked'] = np.hstack(sitesMasked)

    df_bed.to_csv(outputFilePath, sep='\t', index=False, mode='a')

#AI written version. approximately 3x speed up. 
def write_polarized_bed_fast(inputFilePath, outputFilePath, diemTypeObj):
    """
    Fast version of write_polarized_bed - streams data instead of loading everything into memory.
    
    Parameters:
    inputFilePath (str): Path to the original DiEM BED file.
    outputFilePath (str): Path to the output polarized DiEM BED file.
    diemTypeObj (DiemType): DiemType object containing polarity information.
    """
    
    if diemTypeObj.PolByChr is None:
        raise ValueError("DiemType object does not contain polarity information.")
    if inputFilePath is None or outputFilePath is None:
        raise ValueError("Input and output file paths must be provided.")
    if inputFilePath == outputFilePath:
        raise ValueError("Input and output file paths must be different to avoid overwriting.")

    # Prepare site masking data
    sitesMasked = [np.zeros(len(diemTypeObj.posByChr[i]), dtype=int) for i in range(len(diemTypeObj.chrNames))]
    
    if diemTypeObj.siteExclusionsByChr is not None:
        for i in range(len(diemTypeObj.chrNames)):
            if diemTypeObj.siteExclusionsByChr[i] is not None:
                sitesMasked[i][diemTypeObj.siteExclusionsByChr[i]] = 1
    
    # Flatten all the new column data
    nullPolarity_flat = np.hstack(diemTypeObj.initialPolByChr)
    polarity_flat = np.hstack(diemTypeObj.PolByChr)
    DI_flat = np.hstack(diemTypeObj.DIByChr)
    support_flat = np.hstack(diemTypeObj.SupportByChr)
    siteMasked_flat = np.hstack(sitesMasked)
    
    # Stream processing
    with open(inputFilePath, 'r') as f_in, open(outputFilePath, 'w') as f_out:
        
        # Write preamble
        if diemTypeObj.indExclusions is not None:
            masked_inds = '##IndividualsMasked=' + ','.join(diemTypeObj.indExclusions)
            f_out.write(masked_inds + '\n')
        else:
            f_out.write('##IndividualsMasked=None\n')
        
        # Process header line
        header = next(f_in).strip()
        new_header = header + '\tNullPolarity\tPolarity\tDiagnosticIndex\tSupport\tSiteMasked\n'
        f_out.write(new_header)
        
        # Process data lines
        row_idx = 0
        for line in f_in:
            line = line.strip()
            if line:  # Skip empty lines
                new_line = (f"{line}\t{nullPolarity_flat[row_idx]}\t{polarity_flat[row_idx]}\t"
                           f"{DI_flat[row_idx]}\t{support_flat[row_idx]}\t{siteMasked_flat[row_idx]}\n")
                f_out.write(new_line)
                row_idx += 1
