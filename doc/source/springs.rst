Springs
===============================

Overview
^^^^^^^^

Applies a harmonic potential tethering atoms to user-defined coordinates with a potential given by 

.. math::
    U_{\text{spring}} = \frac{1}{2} \sum_{d=x,y,z} k(r_d - r_{d0})^2,

where {k} is a scalar and \mathbf{r}_0=(r_{x0},r_{y0},r_{z0})` is a ``Vector`` which denotes each atom's tether position.  Anisotropic springs including one and two dimensional springs can be created by setting different coefficients for different dimensions.  

By default each atom is tethered to its position when the spring fix is initialized, however arbitrary positions can be set for each atom as shown in the examples.

Springs differ from external potentials in that springs can tether each atom to a different position, while external potentials simulate all atoms within the relevant group in the same external field.


Constructor
^^^^^^^^^^^

.. code-block:: python
    
    FixSpringStatic(state, handle, groupHandle, k, tetherFunc, multiplier)

Arguments

``state``
    Simulation state to apply the fix. Named argument.

``handle``
    A name for the fix. Named argument.

``groupHandle``
    Group of atoms to which the fix is applied.  Named argument.  

``k``
    Spring coefficient for tethers.  Floating point number.  Named argument

``tetherFunc``
    Python function which, when passed an atom, must return a ``Vector`` which is the position to which that atom will be tethered.  Optional.  Defaults to tethering to current position.  Named argument.

``multiplier``
    Scales ``k`` in each dimension.  Can be set to create anisotropic potentials or 2d / 1d springs.  Optional.  Defaults to ``Vector(1, 1, 1)``.  Named argument.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    updateTethers()

No arguments.  Recalculates atom tether positions based on new atom coordinates.  If no ``tetherFunc`` is supplied, tethers are set to atoms' current positions.  If a ``tetherFunc`` is supplied, ``tetherFunc`` is called for each atom in ``groupHandle`` and new positions are determined.

Python Members
^^^^^^^^^^^^^^^^^^^^^^^

**multiplier**

.. code-block:: python

    #makes anisotropic spring that acts with kx = k, ky = 2k, kz = 0
    spring.multiplier = Vector(1, 2, 0)

**k**

.. code-block:: python

    #sets spring constant for tethers
    spring.k = 10

**tetherFunc**

.. code-block:: python

    def myFunc(atom):
        return Vector(round(atom.pos[0]), round(atom.pos[1]), round(atom.pos[2]))
    #sets new tetherFunc for the spring
    spring.tetherFunc = myFunc

    #must call updateTethers to generate new tethers
    spring.updateTethers()



Examples
^^^^^^^^

.. code-block:: python

    #creates fix which tethers all atoms in the group 'substrateAtoms' to their 
    #current positions with k=10
    spring = FixSpringStatic(state, handle='spring1', groupHandle='substrateAtoms', k=10)
    state.activateFix(spring)

.. code-block:: python
    
    def myTetherFunc(atom):
        if atom.pos[0] > 10:
            return Vector(15, atom.pos[1], atom.pos[2]):
        else:
            return atom.pos

    #create a spring which will tether only in the x dimension.  Atoms with x>10 will be tethered to x=15 and all others will be tethered to their original x position.
    spring = FixSpringStatic(state, handle='spring2', groupHandle='all', k=5, tetherFunc=myTetherFunc, multiplier=Vector(1, 0, 0))

    #run the simulation
    integrator = IntegratorVerlet(state)
    integrator.run(1000)

    #change spring constant
    spring.k = 10
    #now spring applies in y dimension as well
    spring.multiplier[1] = 1

    #update atom tethers based on current atom positions.
    spring.updateTethers()

    
