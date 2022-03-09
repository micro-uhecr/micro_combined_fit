# combined_fit: A tool to fit simultaneously UHECR data
Software dedicated to provide a self-consistent modeling framework for fitting simultaneously  energy spectrum, mass composition and arrival direction information 

The main repository is at: https://gitlab.in2p3.fr/micro/micro_combined_fit
The documentation (to be updated soon) is at: https://micro.pages.in2p3.fr/micro_combined_fit/

A mirror is proposed at: https://github.com/micro-uhecr/micro_combined_fit

## Content
The combined_fit project contains the main code in "combined_fit" folder, including:

- Fit.py : 
	main code that performs the fit of  the energy spectrum and  mass composition information.
    
- Propagate.py : 
        main code that allows to see the effect of the extra-galactic propagation for a given choice of the parameters at the source
   
- tensor.py : 
      subclass that handles the reading of the tensor. The tensor is a 4-d matrix which allows to take into account the extra-galactic propagation of the simulated injected fluxes. 
       
- minimizator.py : 
        subclass that handles the minimizator functions 
        
- spectrum.py : 
    subclass that handles the expected and the experimental energy spectrum
    
- mass.py : 
    subclass that handles the expected and the experimental mass composition information. It is possible to plot both lnA, sigma(lnA) and Xmax, sigma(Xmax)
    
- xmax_tools.py : 
        library that contains some tools used in mass.py for the calculation of the expected Xmax.


- constant.py : 
        library that contains some tools used in mass.py for the calculation of the expected Xmax.


In addition to that, in the projects many supplementary folders are provided. They are necessary for running the code:

- Tensor
    It contains an example of tensor, obtained using SimPropv2r4 (arXiv:1705.03729) for five injected masses (A = 1,4,14,28,56)
    
- Data
    Energy spectrum and mass composition data presented at ICRC2019 by the Pierre Auger Observatory.
    
- Catalog
    Catalog used for the evolution in redshift of the sources.
    


## Install instructions
The code is based on python 3, and is meant to be run in its `conda` environment.
The main dependencies are `matplotlib`, `numpy`, `astropy`, `pandas`, `scipy` and `iminuit`.

### conda default environment
```
# clone the project using one of the following options
## 1. IN2P3 gitlab using ssh key 
> git clone git@gitlab.in2p3.fr:micro/micro_combined_fit.git
## 2. IN2P3 gitlab using credentials
> git clone https://gitlab.in2p3.fr/micro/micro_combined_fit.git
## 3. github mirror
> git clone https://github.com/micro-uhecr/micro-combined_fit.git

# create a new conda environment and activate it
> cd micro_combined_fit
> conda env create -f environment.yml
> conda activate micro_combined_fit

# setup current working directory for development
> pip install -e .

# unpack all data files (compressed with zstd)
> ./unpack_data.sh
```

### if you wish to build the API documentation
```
# First install sphinx and the ReadTheDocs theme
> conda install sphinx sphinx_rtd_theme

# Build like this
> cd docs
> make html
```

Open the html documentation in `_build/html/index.html`

### to run the tests and get the coverage
Install pytest
```
> conda install pytest pytest-cov
```

Run coverage
```
./do_cover.sh
```

## Quick start
- Example of propagation of the injected flux for a certain source model
  - via the `Fit.py` python script
  ```
  > python Example/Fit.py
  ```
  - or running the `Propagate.pynb` jupyter notebook 
  ```
  > jupyter notebook &
  ```
  then open `Notebook/Propagate.ipynb`

