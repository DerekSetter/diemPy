# Example Workflow

** USE THE EXAMPLE WORKFLOW **

** We strongly recommend using jupyter notebook in the example_workflow directory of diemPy to get started**

** Please copy this to a personal folder and run it there **

Let <../my_diem_example/> be the path to the copied version of the example_workflow folder that you made.
In a terminal, navigate to that folder, and do:

```sh
conda activate diem
cd ~
jupyter lab
```

Jupyter lab will open in your web browser. From within jupyter lab, on the left hand side you will see the file explorer. Navigate to your personal example folder, open the file workflow_example.ipynb, and follow the tutorial   

When you are ready to run diem on your own data, simply copy the workflow notebook and change the file paths accordingly



# Basic Use Outline 

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

#load the diemtype back in for further analysis
d_smooth = load_DiemType('output.diemtype')
```