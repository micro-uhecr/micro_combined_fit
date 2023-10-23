================================================================
Doing a combined fit of UHECRs spectra
================================================================

Software dedicated to provide a self-consistent modeling framework for fitting simultaneously  energy spectrum, mass composition and arrival direction information

The main repository is at:
https://gitlab.in2p3.fr/micro/micro_combined_fit

The documentation (to be updated soon) is at:
https://micro.pages.in2p3.fr/micro_combined_fit/

A mirror is proposed on github at:
https://github.com/micro-uhecr/micro_combined_fit

For more information on the MICRO UHECR project read:
https://micro-uhecr.github.io/

combined_fit package
------------------------------------

The main components are:

- tensor.py :
      subclass that handles the reading of the tensor. The tensor is a 4-d matrix which allows to take into account the extra-galactic propagation of the simulated injected fluxes.

- minimizer.py :
        subclass that handles the minimizator functions.

- spectrum.py :
    subclass that handles the expected and the experimental energy spectrum.

- mass.py :
    subclass that handles the expected and the experimental mass composition information. It is possible to plot both lnA, sigma(lnA) and Xmax, sigma(Xmax).

- map.py :
        subclass that handles the expected skymaps following the chosen tracer.

- xmax_tools.py :
        library that contains some tools used in mass.py for the calculation of the expected Xmax.


- constant.py & utilities.py :
        libraries that contains some tools used in mass.py for the calculation of the expected Xmax.



Documentation Contents
----------------------

.. toctree::
   :includehidden:
   :maxdepth: 3

   install
   quick_start
   combined_fit
