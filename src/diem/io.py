import pandas as pd
import numpy as np
from . import diemtype as dt

# this is a sketch of a function for importing the bed file data and constructing a diemtype object from it
# uses the output of vcf2diembed version from Mid october that stuart sent us  

# we need another function to read in an 'output' diem bed file so that it handles additional fields like the polarity, flipping the input data if needed, DI, support etc. 

def read_diem_bed(bed_file_path,meta_file_path):
    """
    Reads a DiEM BED file and returns a DiemType object.
    
    Parameters:
    bed_file_path (str): Path to the DiEM BED file.
    meta_file_path (str): Path to the DiEM metadata file.

    Returns:
    DiemType: DiemType object containing the DiEM BED data.
    """

    df_meta = pd.read_csv(meta_file_path, sep='\t')
    chrNames = np.array(df_meta['#Chrom'].values)
    
    #print(chrNames)
    ### For lengths, this is wrong!  just a placeholder. need to chnage to length when this is available ###
    chrLengths = np.array(df_meta['End_diem_input'].values)

    #print(chrLengths)
    sampleNames = np.array(df_meta.columns[4:])
    #print(sampleNames)
    #Assuming first three columns are #Chrom, Start_diem_input, End_diem_input, n(diem_inputs)

    ploidyByChr = []
    for chr in chrNames:
        row = df_meta[df_meta['#Chrom'] == chr]
        ploidy = np.array(row.iloc[0,4:].values, dtype=int)
        ploidyByChr.append(ploidy)

    #print(ploidyByChr)
    column_names = [
            'chrom', 'start', 'end', 'qual', 'ref', 
            'ordered_SAs', 'SNV', 'nVNTs', 
            'exclusion_criterion', 'diem_genotype'
        ]
    
    df_bed = pd.read_csv(bed_file_path, sep='\t', names=column_names,comment='#') 

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
        allele_matrix = [list(s)[1:] for s in df_bed['diem_genotype']]
        allele_matrix = np.array([[map_gt_to_diem(gt) for gt in s] for s in allele_matrix],dtype = np.int8)
        
        allele_matrix = allele_matrix.transpose()  # transpose to shape (n_samples, n_variants)
        DMBC.append(allele_matrix)

    #print(DMBC[0][:,:10])  # print first 10 variants for first chromosome


    return dt.DiemType(DMBC,sampleNames,ploidyByChr,chrNames,positionByChr,chrLengths)


