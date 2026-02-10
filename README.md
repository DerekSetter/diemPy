## diemPy

A Python package for genome polarization and subsequent analyses using the `diem` (Diagnostic Index for the Expectation Maximization) method [1]

## Overview

`diemPy` is a computational tool designed to polarize genomic data for hybrid zone analysis. The package implements an expectation-maximization (EM) algorithm to determine the optimal polarization of genetic markers, enabling researchers to identify and analyze patterns of introgression and hybridization in genomic datasets.



## Key Features
- VCF processing and format conversion
- Automated genome polarization using EM algorithm
- Kernel smoothing and tract length analysis
- Parallel processing support
- Flexible I/O for various genomic formats


## Documentation

The core concepts, installation, and API documentation can be found here:

📚 **[Introduction, Installation, and API Documentation](https://diempy.readthedocs.io/en/latest/intro.html)**

The primary documentation for using `diemPy` is provided as a self-guided tutorial with an example dataset:

**[Tutorial]()
This documentation 
The documentation provides:
- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md) 
- [API Reference](docs/api.md)
- [Example Workflows](docs/workflow_dev_test.ipynb)


## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Author

Derek Setter, Stuart J.E. Baird

## Citation
[1] Baird, S. J. E., Petružela, J., Jaroň, I., Škrabánek, P., & Martínková, N. (2023). Genome polarisation for detecting barriers to geneflow. Methods in Ecology and Evolution, 14, 512–528. https://doi.org/10.1111/2041-210X.14010