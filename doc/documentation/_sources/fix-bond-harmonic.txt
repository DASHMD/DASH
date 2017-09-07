Harmonic Bond Style
====================

Overview
^^^^^^^^

Define a bonding potential energy function corresponding to a harmonic bond style

.. math::
    U_{ij} = \frac{1}{2}k(r - r_0)^2,

where :math:`k, r_0` are parameters that must be set to define the interaction between atoms :math:`i, j` .

Bonds and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Bond types and bonds can be set from within the Python environment with simple invocations. In the syntax that follows, parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing bond type has already been set, it is unneccessary to specify its parameters again when creating a bond. 

.. code-block:: python

    createBond(a,b,k=...,r0=...,type=...)
    setBondTypeCoefs(k=...,r0=...,type=...)

Arguments

``a,b``
    Indices for atoms for which the bond is defined.

``k``
    Spring constant coefficient for harmonic interaction.

``r0``
    Equilbrium bond for harmonic bond interaction.

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for harmonic bond style 
    bondPot = FixBondHarmonic(state,'bondPot')
    
Setting bond type coefficients in python

.. code-block:: python

    #Setting bond types
    bondPot.setBondTypeCoefs(k=100.0,r0=1.0,type=0)

Defining a bond type in python

.. code-block:: python

    #Creating a bond between atoms 1,2 and 2,3
    a=1     # index for atom 1
    b=2     # index for atom 2
    c=3     # index for atom 3
    bondPot.createBond(a,b,type=0)
    # create bond and implicitly create type
    bondPot.createBond(b,c,k=80.0,r0=1.5,type=1) 

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(bondPot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    bond_coeff type k r0
