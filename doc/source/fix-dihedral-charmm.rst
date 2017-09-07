CHARMM/Cosine Dihedral Style
============================

Overview
^^^^^^^^

Define a four-body potential energy function corresponding to the CHARMM/cosine dihedral style

.. math::
    U_{ijkl} = &k\bigl[1+\cos(n\phi - \delta)\bigr],

where :math:`k, n, \delta` are parameters that must be set to define the dihedral interaction between atoms :math:`i, j, k, l` . 

Dihedrals and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Setting dihedral types and dihedrals from within the Python environment is done using simple member functions. In the syntax that follows, parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing dihedral type has already been set, it is unneccessary to specify its parameters again when creating a dihedral for a set of four atoms. 

.. code-block:: python

    setDihedralTypeCoefs(type=...,k=...,n=...,d=...)
    createDihedral(a,b,c,d,type=...,k=...,n=...,d=...)

Arguments 

``a,b,c,d``
    Indices for atoms for which the dihedral is defined.

``k``
    Scale coefficient for interaction.

``n``
    Integer affecting periodicity.

``d``
    Phase shift factor (specified in radians from within python)

``type``
    Integer indicating the dihedral type assignment

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for CHARMM dihedral style 
    torsPot = FixDihedralCHARMM(state,'torsPot')
    
Setting dihedral type coefficients in python

.. code-block:: python

    #Setting dihedral types
    torsPot.setDihedralTypeCoefs(type=0,k=1.,n=1,d=pi)
    torsPot.setDihedralTypeCoefs(type=1,k=0.5.,n=2,d=0.)

Creating a dihedral in python

.. code-block:: python

    #Creating an dihedral between atoms 1,2,3
    a=1     # index for atom 1
    b=2     # index for atom 2
    c=3     # index for atom 3
    d=3     # index for atom 4
    torsPot.createDihedral(a,b,c,d,type=0)
    torsPot.createDihedral(a,b,c,d,type=1)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(torsPot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    dihedral_coeff type k n d

