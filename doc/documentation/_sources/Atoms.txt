Atoms
==============

Overview
^^^^^^^^

Whether you are performing atomisitic or coarse-grained simulations, the ``Atom`` class represents the particles that are interacting with each other in the system.  The ``state.atoms`` attribute is a list of ``Atom`` objects, which can be directly accessed.  Each ``Atom`` stores the dynamic data for that particle - position, velocity, and force - which are updated over the course of the simulation.  It also stores the static properties of the particle - mass, charge, atom type, and id - which do not change during the simulation.

Adding atoms to the simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atoms can be introduced into the simulation by adding them directly to the ``state`` or by using the ``InitializeAtoms`` tools.  The first method is described below and the second is discussed in the next section.

**state.addAtom(** handle, pos, q **)**

Adds an atom to the ``state``.  The atom is only added if a valid atom handle is supplied.

**Arguments**

``handle``: The string representation of the corresponding atom type in the ``AtomParams`` object.

``pos``: A Vector specifying the position of the new atom.

``q``: The charge for this atom (optional).

**Returns**

``id``: The id of the newly added atom. Returns ``-1`` if the atom could not be added (invalid atom type).


**Example**

The following code demonstrates inserting three charged atoms into the simulation 

.. code-block:: python

	#Suppose our AtomParams object has atom types 'spc1' and 'spc2'
    
	#create new positions for the atoms
	
	#add in the atoms
	state.addAtom(handle='spc1', pos=Vector(0, 0, 0), q=-0.1)
	state.addAtom(handle='spc2', pos=Vector(2, 0, 0), q= 0.2)
	state.addAtom(handle='spc2', pos=Vector(2, 2, 0), q=-0.1)
	

Accessing and updating atom data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class** Atom

**Attributes**

The following atributes can be accessed for any given ``Atom`` object:

``pos``: Vector containing the particle's position (read/write).

``vel``: Vector containing the particle's veloctiy (read/write).

``force``: Vector containing the sum of all forces acting on the particle (read/write).

``groupTag``: A list of ids corresponding the atom groups this particle is associated with (read/write).

``mass``: The mass of the particle (read/write).

``q``: The charge of the atom (read/write).

``type``: The id of the corresponding atom type in the ``AtomParams`` object (read-only).  This is set on atom initialization (discussed elsewhere).

``kinetic``: The kinetic energy of this particle, calculated accoridng to the classical formula K = (m*v^2)/2 (read-only).

``isChanged``: A boolean flag which is set to ``True`` when an integrator updates the position, velocity, or force on the particle (read/write).


**Example**

The following example illustrates the syntax used to access and update atom data. In the course of initializing a simulation, it is common to programmatically assign starting positions, masses, and charges. For instance, to initialize a water molecule, one could write

.. code-block:: python
	
		#Suppose our state contains three atoms
	
		#create new positions for the atoms
		pos1 = Vector(1,1,1)
		pos2 = Vector(1,3,5)
		pos3 = Vector(5,7,9)
	
		#update the atom positions
		state.atoms[0].pos = pos1
		state.atoms[1].pos = pos2
		state.atoms[2].pos = pos3
	
		#update the atom charges
		state.atoms[0].q = -1
		state.atoms[1].q = 0.5
		state.atoms[2].q = 0.5



Setting atom parameters
^^^^^^^^^^^^^^^^^^^^^^^

The ``AtomParams`` object contains a directory of the atom handles/types found in the simulation along with masses and atomic numbers; handles are text identifiers for atom types (see example below).  In order to add an atom, an atom type is needed, which must also be specified in ``AromParams``.  Fixes to set interaction parameters between various atom types interface with ``AtomParams``.

Class AtomParams
""""""""""""""""

**Attributes**

The following attributes and methods of the ``AtomParams`` object are available:

``handles``: A list of all the atom species handles (text names) in the simulation (read-only).

``numTypes``: The number of atom types in the simulation (read-only).

``masses``: A list of the atom species masses in the simulation (read/write).

Methods
"""""""

**atomParams.addSpecies(** handle, mass, atomicNum **)**

`Arguments`:
	
``handle``: The handle for the new species.

``mass``: The mass for this species.

``atomicNum``: The atomic number for the species (optional).

`Returns`:

``id``: The atom type id (integer) for the newly added species.


**atomParams.typeFromHandle(** handle **)**

`Arguments`:
	
``handle``: The handle (text) of a given species.

`Returns`:

``id``: The atom type id (integer) corresponding to the ``handle``.

**atomParams.setValues(** handle, mass, atomicNum **)**

Updates the mass and/or atomic number of a given species.

`Arguments`:
	
``handle``: The handle for the species to be updated.

``mass``: The new mass for the species (optional).

``atomicNum``: The new atomic number for the species (optional).

`Returns`:

None.

**Example**

The following example illustrates the syntax used to set atom parameters and update them.

.. code-block:: python
	
	#Set up the parameters for a carbon atom
	state.atomParams.addSpecies(handle='myC', mass=12)

	#update the mass and atomic number
	state.atomParams.setValues(handle='myC', mass=12.0107, atomicNum=6)







Tools for initializing atoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``InitializeAtoms`` class provides a number of tools for initializing atom positions and velocities.


Randomly populate the simulation box
""""""""""""""""""""""""""""""""""""

**InitializeAtoms.populateRand(** state, bounds, handle, n, distMin **)**

Randomly adds `n` atoms to the simulation box within the given bounds, subject to a minimum allowable distance between atoms.

**Arguments**

``state``: The state to add atoms to.

``bounds``: The bounds within which to add the atoms.

``handle``: The string representation of the atom type to be added.

``n``: The number of atoms to add.

``distMin``: The minimum allowable distance between atoms.

**Returns**

None.


**Example**

The following code demonstrates the addition of some Lennard-Jones particles using this method.

.. code-block:: python
	
	#Set up the parameters for a basic LJ particle
	state.atomParams.addSpecies(handle='myLJ', mass=1)
	ljcut = FixLJCut(state, handle='ljcut')
	state.activateFix(ljcut)
	ljcut.setParameter(param='eps', handleA='myLJ', handleB='myLJ', val=1)
	ljcut.setParameter(param='sig', handleA='myLJ', handleB='myLJ', val=1)

	#set the bounds for a 5x5x5 box
	state.bounds = Bounds(state, lo=Vector(0, 0, 0), hi=Vector(5, 5, 5))
	
	#Randomly add a bunch of atoms, this gives a reduced density of about 0.5
	InitializeAtoms.populateRand(state, bounds=state.bounds, handle='myLJ', n=64, distMin = 0.75)


Initialize atom velocities
""""""""""""""""""""""""""

**InitializeAtoms.initTemp(** state, handle, temp **)**

Initializes the atoms in a given group to the desired temperature with center-of-mass motion removed.

**Arguments**

``state``: The simulation state.

``bounds``: The bounds of the volume in space to be populated.

``handle``: The group name to be set to the desired temperature.

``temp``: The desired temperature.

**Returns**

None.


**Example**

The following code demonstrates the i of some Lennard-Jones particles using this method.

.. code-block:: python

	#Set up the parameters for a basic LJ particle
	state.atomParams.addSpecies(handle='myLJ', mass=1)
	ljcut = FixLJCut(state, handle='ljcut')
	state.activateFix(ljcut)
	ljcut.setParameter(param='eps', handleA='myLJ', handleB='myLJ', val=1)
	ljcut.setParameter(param='sig', handleA='myLJ', handleB='myLJ', val=1)

	#set the bounds for a 5x5x5 box
	state.bounds = Bounds(state, lo=Vector(0, 0, 0), hi=Vector(5, 5, 5))
	
	#Randomly add a bunch of atoms, this gives a reduced density of about 0.5
	InitializeAtoms.populateRand(state, bounds=state.bounds, handle='myLJ', n=64, distMin = 0.75)
	
	#Initialize the velocities to a reduced temperature of 0.5
	InitializeAtoms.initTemp(state, 'all', 0.5)
	

Water molecules
^^^^^^^^^^^^^^^

DASH includes utilities for creating water molecules based on TIP3P, TIP4P, and TIP4P/2005 models as well as TIP3P and TIP4P for long range electrostatics solvers.  These functions are included it ``water.py`` within the ``util_py`` folder.  All methods return a :doc:`Molecule</molecule>` object, which may then be added to the relevent rigid fix.  Methods include ``create_TIP3P`` (Jorgensen JCP, 1983), ``create_TIP3P_long`` (Price, JCP, 2004), ``create_TIP4P`` (Jorgensen JCP, 1983), ``create_TIP4P_long`` (Price, JCP, 2004), ``create_TIP4P_2005`` (Vega, JCP, 2005).  Note that
Lennard-Jones parameters must be initialized by the user.  

.. code-block:: python

    import sys
    sys.path.append('/path/to/util_py/')
    import water

    #returns Molecule object
    tip3p = water.create_TIP3P()


    myRigidFix.createRigid(tip3p)


    

Deleting atoms
^^^^^^^^^^^^^^

Atoms can also be deleted from the ``state``.

Atoms can be introduced into the simulation by adding them directly to the ``state`` or by using the ``InitializeAtoms`` tools.  The first method is described below and the second is discussed elsewhere.

**state.deleteAtom(** a **)**

Deletes the specified atom from the ``state`` and all associated fixes.

**Arguments**

``a``: An atom object


**Returns**

``bool``: A boolean.  ``True`` means the atom was successfully deleted.


**Example**

The following code demonstrates this method of removing atoms into the simulation using the water example from above:

.. code-block:: python

	#Suppose our AtomParams object has atom types 'spc1'
    
	#create new positions for the atom
	posO = Vector(1,1,1)
	
	#add the atoms
	state.addAtom('spc1', posO, -0.834)

	#delete the atoms
	state.deleteAtom(state.atoms[0])
	

