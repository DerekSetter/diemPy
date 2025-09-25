# Installation

To install `diem`, first set up a new conda environment:

```sh
conda create -n diem python=3.11
conda activate diem
```

Then install the require dependencies
```sh
conda install -c conda-forge numpy pandas numba
```

Finally, install `diem` using pip. From within your conda environment, and in the diem directory (where the pyproect.toml file is located) run :

```sh
python -m pip install -e .
```

