.. _install:

Installing combined_fit
=======================

At some point you will be able to install `combined_fit` with `pip` or from source.


Installing with pip
-------------------

This is will work one day, believe me.

.. code-block:: bash

    pip install combined_fit



Installing from source
-----------------------

Since `combined_fit` is all python code, this is pretty easy.

.. code-block:: bash

    git clone git@gitlab.in2p3.fr:micro/combinedfit.git combined_fit
    cd combined_fit
    conda env create -f environment.yml
    conda activate combined_fit
    pip install -e .
