Getting Starting
^^^^^^^^^^^^^^^^

Here we give a simple DASH script to get things started.


This script simply creates two atoms which interact via a Lennard-Jones potential.

.. code-block:: python

    import sys.path
    sys.path.append('/the/path/to/DASH.so/')

    from DASH import *

    #initialize simulation state
    #Lennard-Jones units are the default
    state = State()
    
    #Now we create an atom type
    state.atomParams(handle='spc1', mass=1)

    #initialize system bounds
    state.bounds = Bounds(state, lo=Vector(0, 0, 0), hi=Vector(100, 100, 100))

    #set cutoff radius
    state.rCut = 2.5
    #set neighborlist padding
    state.padding = 0.5

    #add two atoms
    state.addAtom(handle='spc1', pos=Vector(1, 1, 1))
    state.addAtom(handle='spc1', pos=Vector(3, 1, 1))

    #Their properties can be accessed from within the python script
    print state.atoms[1].pos
    #prints out (3, 1, 1)
    
    #Set the velocity of an atom
    state.atoms[0].vel = Vector(0.1, 0, 0)


    #Initialize Lennard-Jones parameters
    nonbond = FixLJCut(state, 'cut')
    #set sigma and epsilon
    nonbond.setParameter('sig', 'spc1', 'spc1', 1)
    nonbond.setParameter('eps', 'spc1', 'spc1', 1)

    #turn on fix
    state.activateFix(nonbond)

    #create an integrator.
    integrator = IntegratorVerlet(state)
    #run the simulation
    integrator.run(1000)

    print state.atoms[0].pos, state.atoms[1].pos


We could bond the atoms together as well.

.. code-block:: python
    
    bondHarmonic = FixBondHarmonic(state, handle='bond')
    bondHarmonic.createBond(state.atoms[0], state.atoms[1], k=1, r0=2)
    state.activateFix(bondHarmonic)
    integrator.run(1000)


These examples show the basics of how one can interact with DASH through Python.  For more information on functionality and interface, see the remainder of the documentation.

