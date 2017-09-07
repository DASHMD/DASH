CHARMM Angle Style
====================

Overview
^^^^^^^^

Define a three-body potential energy function corresponding to the CHARMM angle style

.. math::
    U_{ijk} = \frac{1}{2}K(\theta - \theta_0)^2 + \frac{1}{2}K_{\text{ub}}(r-r_{\text{ub}})^2,

where :math:`K, \theta_0, K_{\text{ub}}, r_{\text{ub}}` are coefficients that must be set to define the interaction between atoms :math:`i, j, k` .

Angles and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Setting angle types and angles from within the Python environment is done with simple invocations. Parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing angle type has already been set, it is unneccessary to specify its parameters again when creating an angle for three atoms. 

.. code-block:: python

    setAngleTypeCoefs(k=...,theta0=...,kub=...,rub=...,type=...)
    createAngle(a,b,c,k=...,theta0=...,kub=...,rub=...,type=...)

Arguments 

``a,b,c``
    Indices for atoms for which the angle is defined.

``k``
    Spring constant coefficient for harmonic interaction.

``theta0``
    Equilbrium angle for harmonic angle interaction (specified in radians).

``kub``
    Urey-Bradley coefficient for 1-3 interaction.

``rub``
    Equilibrium Urey-Bradley separation distance.

``type``
    Integer indicating the angle type assignment

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for CHARMM angle style 
    anglePot = FixAngleCHARMM(state,'anglePot')
    
Setting angle type coefficients in python

.. code-block:: python

    #Setting angle types
    anglePot.setAngleTypeCoefs(k=100.0,theta0=2*pi/3.,type=0)

Defining an angle type in python

.. code-block:: python

    #Creating an angle between atoms 1,2,3
    a=1     # index for atom 1
    b=2     # index for atom 2
    c=3     # index for atom 3
    d=3     # index for atom 4
    anglePot.createAngle(a,b,c,type=0)
    # create angle and implicitly create type
    anglePot.createAngle(b,c,d,k=80.0,theta0=pi*100./180.,kub=10.0,rub=1.0,type=1)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(anglePot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    angle_coeff type k theta0 kub rub

