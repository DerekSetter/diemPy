# # #this file marks this diem file as a python package
# # from . import diplotype
# # from . import polarize
# # from . import kernel_smoothing_functions
# # from . import haplotypes
# # from . import diem_utils
    
# #import pydoc 
# import pydoc
# from . import diplotype
# from . import polarize


# pydoc.doc(diplotype)
# pydoc.doc(polarize)
# # pydoc.doc(diem.polarize)
# # pydoc diem.kernel_smoothing_functions
# # pydoc diem.haplotypes
# # pydoc diem.diem_utils   

from . import diemtype
from . import polarize
from . import contigs
from . import smooth
from . import tests
from . import utils
from . import io
from . import plots

# Import key functions for direct access
# I/O functions
from .io import read_diem_bed
from .io import write_polarized_bed
from .io import update_ploidy

# DiemType object handling
from .diemtype import save_DiemType
from .diemtype import load_DiemType

# Main analysis functions
from .polarize import run_em_parallel
from .polarize import run_em_linear

# Smoothing functions
from .smooth import laplace_smooth_multiple_haplotypes
from .smooth import laplace_smooth_multiple_haplotypes_parallel

# Contig analysis
from .contigs import build_contig_matrix
from .contigs import export_contigs_to_ind_bed_files

# Utility functions
from .utils import plot_painting
from .utils import plot_painting_with_positions
from .utils import characterize_markers
from .utils import count_site_differences

# Make DiemType class directly accessible
from .diemtype import DiemType