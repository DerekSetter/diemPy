# Installation

To install `diempy`, first create a new conda environment then use pip to install the package

```sh
conda create -n diempy python=3.11
conda activate diempy
python -m pip install diempy
conda install -c conda-forge jupyterlab
```

`diempy` can be installed as a package and used in any python script. This means that the big tasks like polarization can be offloaded to a computing cluster. However, for many datasets, `diempy` can run on a modern laptop. We recommend using `diempy` and exploring the data within a jupyter notebook, and jupyter lab comes pre-installed.


