# Installation

To install `diem`, first download or clone the repository.

Then, set up a new conda environment:

```sh
conda create -n diem python=3.11
conda activate diem
```

install `diem` using pip. From within your conda environment and in the diem directory (where the pyproect.toml file is located) run :


```sh
python -m pip install .
```

`diem` can be installed as a package and used in any python script. This means that the big tasks like polarization can be offloaded to a computing cluster. However, for many datasets, `diem` can run on a modern laptop. We recommend using `diem` and exploring the data within a jupyter notebook, and we provide an example workflow as a starting point. To install jupyter lab:

```sh
conda install -c conda-forge jupyterlab
```  
