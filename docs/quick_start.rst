.. _quick_start:

Quick start guide
*****************

Run a simulation
================

It's easy to setup and run a quick simulation: once files are unzipped (./unpack_data.sh)
it is possible to run one of the following example.

Visualization
=============
Including some plots?

Notebooks
=========
In the notebook folder we provide three examples:

- Propagate
- Fit
- Maps

Scripts
=======
`Example/Propagate.py`
---------------------------
Script used to propagate the UHECR flux corresponding to a given parameters at the sources, such as the spectral index, the rigidity cutoff and the relative abundance of the different atomic species.

.. code-block::

    > python Example/Propagate.py

`Example/Fit.py`
----------------------------
Script to search the best parameters at the source which describes both spectrum and composition at Earth.

.. code-block::

    > python  Example/Fit.py

`Example/Map.py`
-------------------------------
Script to produce the skymaps corresponding to the best fit scenario, assuming a given tracer for the UHECR production rate (SFR o SMD).

.. code-block::

    > python Example/Map.py
