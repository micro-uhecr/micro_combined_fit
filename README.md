# combined_fit: A tool to fit simultaneously UHECR data
Software dedicated to provide a self-consistent modeling framework for fitting simultaneously  energy spectrum, mass composition and arrival direction information 
                                                            
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
    


## Environment
The code can be compatible using python 3.


#### Dependencies 
The software depends on standard python packages:
- astropy
- numpy
- scipy
- matplotlib
- pandas

#### Encountered issues
- Depending on the python version, the automatic installation of healpy does not work. As healpy is optional, it was removed from the dependencies and healpy can be installed independently if necessary.

- The automatic installation of dependencies is sometimes misbehaving. In such case, you may just install the required packages independently:

```
> conda install astropy
> conda install numpy
> conda install scipy
> conda install matplotlib
> conda install pandas
> conda install numba
```

## Install instructions (for development)

### conda
```
> git clone https://gitlab.in2p3.fr/micro/combinedfit.git combined_fit
> cd combined_fit
> conda env create -f environment.yml
> conda activate combined_fit
> pip install -e .
```

### pip requirements first
```
> conda create -n combined_fit python=3.9
> conda activate combined_fit
> git clone https://gitlab.in2p3.fr/micro/combinedfit.git combined_fit
> cd combined_fit
> pip install -r requirements.txt
> pip install -e .
```

### just pip
```
> conda create -n combined_fit python=3.9
> conda activate combined_fit
> git clone https://gitlab.in2p3.fr/micro/combinedfit.git combined_fit
> cd combined_fit
> pip install -e .
```

### to build the doc
First install sphinx and the ReadTheDocs theme
```
> conda install sphinx sphinx_rtd_theme
```

Build like this
```
cd docs
make html
```

Open the html documentation in `_build/html/index.html`

### to run the tests
Install pytest
```
> conda install pytest pytest-cov
```

Run coverage
```
./do_cover.sh
```


