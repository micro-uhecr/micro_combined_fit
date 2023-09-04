.. _install:

Installing combined_fit
=======================

At some point you will be able to install `combined_fit` with `pip` or from source.


Installing with pip
-------------------

Not implemented yet.

.. code-block:: bash

    pip install combined_fit



Installing from source
-----------------------

The code is based on python 3, and is meant to be run in its `conda` environment.
The main dependencies are `matplotlib`, `numpy`, `astropy`, `pandas`, `scipy` and `iminuit`.

A recent version of conda is needed (>22.11) so that one can activate mamba as a dependency solver.

.. code-block:: bash

  ### conda default environment
  ```
  # clone the project using one of the following options
  ## 1. IN2P3 gitlab using ssh key
  > git clone git@gitlab.in2p3.fr:micro/micro_combined_fit.git
  ## 2. IN2P3 gitlab using credentials
  > git clone https://gitlab.in2p3.fr/micro/micro_combined_fit.git
  ## 3. github mirror
  > git clone https://github.com/micro-uhecr/micro-combined_fit.git

  # install and activate mamba as a solver (as needed)
  > conda install -n base conda-libmamba-solver

  # create a new conda environment and activate it
  > cd micro_combined_fit
  > conda config --set solver libmamba
  > conda env create -f environment.yml
  > conda activate micro_combined_fit

  # setup current working directory for development
  > pip install -e .
