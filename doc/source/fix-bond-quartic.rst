Quartic Bond Style
====================

Overview
^^^^^^^^

Define a bonding potential energy function corresponding to a quartic bond style

.. math::
    U_{ij} = k_2(r - r_0)^2 + k_3(r-r_0)^3 + k_4(r-r_0)^4,

where :math:`k_2, k_3, k_4, r_0` are parameters that must be set to define the interaction between atoms :math:`i, j` .

Bonds and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Bond types and bonds can be set from within the Python environment with simple invocations. In the syntax that follows, parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing bond type has already been set, it is unneccessary to specify its parameters again when creating a bond. 

.. code-block:: python

    createBond(a,b,k2=...,k3=...,k4=...,r0=...,type=...)
    setBondTypeCoefs(k2=...,k3=...,k4=...,r0=...,type=...)

Arguments

``a,b``
    Indices for atoms for which the bond is defined.

``k2``
    Spring constant coefficient for quadratic coupling.

``k3``
    Spring constant coefficient for cubic coupling.

``k4``
    Spring constant coefficient for quartic coupling.

``r0``
    Equilbrium bond for bond interaction.

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for quartic bond style 
    bondPot = FixBondQuartic(state,'bondPot')
    
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
    bondPot.createBond(b,c,k2=80.0,k3=-500.0,k4=1000.0,r0=1.5,type=1) 

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(bondPot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    bond_coeff type r0 k2 k3 k4
