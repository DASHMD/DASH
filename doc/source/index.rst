.. Dash documentation master file, created by
   sphinx-quickstart on Tue Apr 25 15:12:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DASH's documentation!
================================


DASH is a GPU-accelerated molecular dynamics software package designed for both atomistic and coarse grained simulations.  DASH is controlled via a powerful python interface, which gives the user full control to set up, modify, and analyze simulations all within one script.  Using efficient parallel algorithms, DASH also provides the facility to compute arbitrarly complex user-defined types of data while maintaining high performance.  

DASH is specifically optimized for running with the the advanced sampling package SSAGES, allowing up to double the performance of leading competitors.  With plain molecule dynamics, performance in DASH is comparable to the fastest codes available. 

DASH supports several water models including TIP3P, TIP4P (and variants), as well as the E3B model, a fast and robust 3-body water model parameterized to capture much of the water phase diagram (Kumar, JPC, 2008).  

In addition to these features, DASH provides all the standard functionality users have come to expect from a batteries-included molecules dynamics package, including a variety of thermostats, barostats, elegant methods for dealing with molecules, long range charge computation, multiple force fields, and much more.


.. DASH uses the SETTLE algorithm (Miyamoto, JCC, 1992) to efficiently model water molecules.  In addition to standard water potentials such as TIP3P, TIP4P and similar, DASH includes the new E3B water model (Tainter, JCP, 2011), which has been parameterized to accurately capture much of the water phase diagram.

Getting started

.. toctree::
   :maxdepth: 1

   compiling
   getting-started

    

Core functionality 

.. toctree::
   :maxdepth: 2
   
   state
   data-recording
   writing-trajectories
   reading-trajectories
   
   Atoms
   Bounds
   Units

   python-operation
   molecule

Potentials

.. toctree::
   :maxdepth: 2

   fix-bond-harmonic
   fix-bond-fene
   fix-bond-quartic
   fix-angle-harmonic
   fix-angle-charmm
   fix-angle-cosinedelta
   fix-dihedral-opls
   fix-dihedral-charmm
   fix-pair-LJ
   fix-pair-LJFS
   fix-pair-TICG
   fix-charge-DSF
   fix-charge-Ewald
   fix-wall-LJ126
   fix-wall-harmonic

External potentials:

.. toctree::
   :maxdepth: 2

   fix-external-harmonic
   fix-external-quartic
   springs

Thermostats and Barostats:

.. toctree::
   :maxdepth: 2

   fix-NoseHoover
   fix-pressure-Berendsen
   fix-Langevin
   fix-NVT-Andersen
   fix-NVT-rescale

Integrators:

.. toctree::
   :maxdepth: 2
   
   integrator-Verlet
   integrator-relax

   
Utilities and external functionality

.. toctree::
   :maxdepth: 2

   lammps-reader
   ssages




Indices and tables
==================

* :ref:`search`

