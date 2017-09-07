OPLS/Cosine Series Dihedral Style
=================================

Overview
^^^^^^^^

Define a four-body potential energy function corresponding to the OPLS/cosine series dihedral style

.. math::
    U_{ijkl} = &\frac{k_1}{2}\bigl[1+\cos(\phi)\bigr]+ \frac{k_2}{2}\bigl[1-\cos(2\phi)\bigr]+ \\
               &\frac{k_3}{2}\bigl[1+\cos(3\phi)\bigr]+ \frac{k_4}{2}\bigl[1-\cos(4\phi)\bigr],

where :math:`k_1, k_2, k_3, k_4` are coefficients that must be set to define the dihedral interaction between atoms :math:`i, j, k, l` .

Dihedrals and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Setting dihedral types and dihedrals from within the Python environment is done using simple member functions. In the syntax that follows, parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing dihedral type has already been set, it is unneccessary to specify its parameters again when creating a dihedral for a set of four atoms. 

.. code-block:: python

    setDihedralTypeCoefs(type=...,coefs=[..,..,..,..])
    createDihedral(a,b,c,d,type=...,coefs=[..,..,..,..])

Arguments 

``a,b,c,d``
    Indices for atoms for which the dihedral is defined.

``coefs``
    List of coefficients corresponding to :math:`k_1, k_2, k_3, k_4` . 

``type``
    Integer indicating the dihedral type assignment

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for OPLS dihedral style 
    torsPot = FixDihedralOPLS(state,'torsPot')
    
Setting dihedral type coefficients in python

.. code-block:: python

    #Setting dihedral types
    torsPot.setDihedralTypeCoefs(type=0,coeffs=[1.0, -0.5, 0.25,0.0])

Defining a dihedral in python

.. code-block:: python

    #Creating an dihedral between atoms 1,2,3,4
    a=1     # index for atom 1
    b=2     # index for atom 2
    c=3     # index for atom 3
    d=3     # index for atom 4
    torsPot.createDihedral(a,b,c,d,type=0)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(torsPot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    dihedral_coeff type k1 k2 k3 k4

