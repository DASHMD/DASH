FENE Bond Style
====================

Overview
^^^^^^^^

Define a bonding potential energy function corresponding to a FENE (finite extensible nonlinear elastic) bond style

.. math::
    U_{ij} = -&\frac{1}{2}kr_0^2 \ln\left[(1 - \bigl(\frac{r}{r_0}\bigr)^2\right]+ \\
              &4\epsilon\left[ \bigl(\frac{\sigma}{r}\bigr)^{12} - \bigl(\frac{\sigma}{r} \bigr)^6\right] + \epsilon

where :math:`k, r_0, \epsilon, \sigma` are parameters that must be set to define the interaction between atoms :math:`i, j` . The second term is set to zero if :math:`r > 2^{1/6}\sigma`.

Bonds and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Bond types and bonds can be set from within the Python environment with simple invocations. In the syntax that follows, parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing bond type has already been set, it is unneccessary to specify its parameters again when creating a bond. 

.. code-block:: python

    createBond(a,b,k=...,r0=...,eps=...,sig=...,type=...)
    setBondTypeCoefs(k=...,r0=...,eps=...,sig=...,type=...)

Arguments

``a,b``
    Indices for atoms for which the bond is defined.

``k``
    Scale coefficient for attractive FENE interaction.

``r0``
    Maximum bond extent.

``eps``
    Repulsive interaction energy for FENE potential.

``sig``
    Distance/size parameter for repulsive part of FENE potential. 

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for FENE bond style 
    bondPot = FixBondFENE(state,'bondPot')
    
Setting bond type coefficients in python

.. code-block:: python

    #Setting bond types
    bondPot.setBondTypeCoefs(k=30.0,r0=1.5,eps=1.0,sig=1.0,type=0)

Defining a bond type in python

.. code-block:: python

    #Creating a bond between atoms 1,2
    a=1     # index for atom 1
    b=2     # index for atom 2
    bondPot.createBond(a,b,type=0)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(bondPot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    bond_coeff type k r0 eps sig
