Harmonic Wall Potential
=======================

Overview
^^^^^^^^
Implements a harmonic potential energy wall at a specified origin ``origin`` with force in the vector direction ``forceDir``.  The cut off distance is specified by the ``dist`` keyword.  The spring constant associated with the wall is denoted by the ``k`` parameter.  The wall potential has a potential energy function given by 

.. math:: 
    U_{\text{wall}} = \frac{1}{2} k (r - r_{0})^2


Constructor
^^^^^^^^^^^
.. code-block:: python

    FixWallHarmonic(state,handle,groupHandle,origin,forceDir,dist,k)



Arguments

``state``
    Simulation state to apply the fix.

``handle``
    A name for the object.  String type.

``groupHandle``
    Group of atoms to which the fix is applied.  String type. 

``origin``
    Point of origin for the wall.  Vector type.   

``forceDir``
    The direction in which the force is to be applied.  Vector type.
``dist``
    The cutoff for the potential.  Float type.

``k``
    The spring constant associated with the wall.  Float type.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

    Member data of the FixWallHarmonicpotential may be modified directly via the Python interface; namely, 'k', 'dist', 'forceDir', and 'origin' keywords are directly accessible from an instance of FixWallHarmonic.

    To modify any these parameters, simply assign them a new value of an appropriate type.

Examples
^^^^^^^^
Example 1: Creating an instance of FixWallHarmonic

.. code-block:: python

    # create a simulation state to which we will add the fix
    state = state()
    
    # set the bounds of the state
    state.bounds = Bounds(state, lo=Vector(0,0,0), hi=Vector(30,30,30))

    # put the wall at (0,0,0)
    origin = Vector(0,0,0)

    # have the wall be acting in the +x direction
    forceDir = Vector(1,0,0)

    # cutoff distance of 15 units
    dist = 15

    # set a spring constant k = 2.5
    k = 2.5

    # create an instance of the fix
    fixWall = FixWallHarmonic(state,"wall","all",origin,forceDir,dist,k)

    # activate the fix
    state.activateFix(fixWall)

Example 2: Modifying the force constant after instantiation

.. code-block:: python

    # increase the spring constant to k = 3.5
    fixWall.k = 3.5
