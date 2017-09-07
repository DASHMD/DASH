LJ Wall Potential
=================

Overview
^^^^^^^^
    Implements a Lennard-Jones 12-6 potential well associated with a boundary of the simulation cell for simulation of non-periodic dimensions.  The form of the potential is given by


.. math::
   V(r_{ij}) =  \left[\begin{array}{cc} 4 \varepsilon \left( \left(\frac{\sigma}{r_{ij}}\right)^{12} -
                    \left(\frac{\sigma}{r_{ij}}\right)^{6}\right),& r<r_{\rm cut}\\
                    0, & r\geq r_{\rm cut}
                    \end{array}\right.


where :math:`r_{ij}` is the distance between a particle and the wall, :math:`\varepsilon, \sigma` are Lennard-Jones potential parameters, and :math:`r_{\rm cut}` is cutoff distance.

Constructor
^^^^^^^^^^^
.. code-block:: python

    FixWallLJ126(state,handle,groupHandle,origin,forceDir,dist,sigma,epsilon)



Arguments

``state``
    Simulation state to which the fix is applied.

``handle``
    Name of the fix.  String type.

``groupHandle``
    Group of atoms to which the fix is applied.  String type.

``origin``
    Location of the wall.  Vector type.

``forceDir``
    Direction in which the wall potential is applied.  Vector type.

``dist``
    Cutoff distance for the potential.  Float type.

``sigma``
    Lennard-Jones :math: `\sigma` parameter associated with the wall.  Float type.

``epsilon``
    Lennard-Jones :math: `epsilon` parameter associated with the wall.  Float type.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
    Member data of the FixWallLJ126 potential may be modified directly via the Python interface; namely, 'sigma', 'epsilon', 'dist', 'forceDir', and 'origin' keywords are directly accessible from an instance of FixWallLJ126.

    To modify any these parameters, simply assign them a new value of an appropriate type.


Examples
^^^^^^^^

Example 1: creating an instance of FixWallLJ126

.. code-block:: python

    # create a simulation state to which we will add the fix
    state = State()

    # set the bounds of the state
    state.bounds = Bounds(state, lo=Vector(0,0,0), hi=Vector(30,30,30))

    # put the wall at (0,0,0)
    origin = Vector(0,0,0)

    # have the wall be acting in the +x direction
    forceDir = Vector(1,0,0)

    # cutoff distance of 15 units
    dist = 15

    # sigma and LJ
    sigma = 2.4
    epsilon = 1.0

    # create an instance of the fix
    fixWall = FixWallLJ126(state,"ljwall","all",origin,forceDir,dist,sigma,epsilon)

    # activate the fix
    state.activateFix(fixWall)

Example 2: Modifying the LJ constants of the above wall

.. code-block:: python

    # referring to the instances created above..
    # change the sigma parameter
    fixWall.sigma=3.0



