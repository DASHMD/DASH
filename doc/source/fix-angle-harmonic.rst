Harmonic Angle Style
====================

Overview
^^^^^^^^

Define a three-body potential energy function corresponding to a harmonic angle style

.. math::
    U_{ijk} = \frac{1}{2}K(\theta - \theta_0)^2,

where :math:`K, \theta_0` are parameters that must be set to define the interaction between atoms :math:`i, j, k` .

Angles and types can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Angle types and angles can be set from within the Python environment with simple invocations. In the syntax that follows, parameters with ``=`` may be given in any order and are also optionally specified. For example, if an existing angle type has already been set, it is unneccessary to specify its parameters again when creating an angle between three atoms. 

.. code-block:: python

    createAngle(a,b,c,k=...,theta0=...,type=...)
    setAngleTypeCoefs(k=...,theta0=...,type=...)

Arguments

``a,b,c``
    Indices for atoms for which the angle is defined.

``k``
    Spring constant coefficient for harmonic interaction.

``theta0``
    Equilbrium angle for harmonic angle interaction (specified in radians if in python).

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #Add Fix for harmonic angle style 
    anglePot = FixAngleHarmonic(state,'anglePot')
    
Setting angle type coefficients in python

.. code-block:: python

    #Setting angle types
    anglePot.setAngleTypeCoefs(k=100.0,theta0=2.*pi/3.0,type=0)

Defining an angle type in python

.. code-block:: python

    #Creating an angle between atoms 1,2,3
    a=1     # index for atom 1
    b=2     # index for atom 2
    c=3     # index for atom 3
    d=3     # index for atom 4
    anglePot.createAngle(a,b,c,type=0)
    # create angle and implicitly create type
    anglePot.createAngle(b,c,d,k=80.0,theta0=pi*100.0/180.0,type=1) 

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(anglePot)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    angle_coeff type k theta0 
