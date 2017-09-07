Recording Data
==============

Overview
^^^^^^^^

Several types of data can be recorded using GPU-accelerated methods in DASH, including energies on an aggregate or fine-grained bases, temperature, kinetic energy, pressture, virial coefficients, volume, and simulation boundaries.  Other data types can be recorded via :doc:`Python Operation<python-operation>` with minimal performance overhead.

Recording data
^^^^^^^^^^^^^^

The ``dataManager`` member of ``State`` handles data recording.  To record temperature of all atoms, for instance, one would write

.. code-block:: python

    #record temperature every 100 turns
    temperatureData = state.dataManager.recordTemperature(interval=100)
    
    integrater.run(10000)

    #print python list of recorded temperatures
    print temperature.vals
    #print python list of turns on which these values were recorded
    print temperature.turns

and as shown, access the recorded values through the ``vals`` member and the turns on which they were recorded through the ``turns`` member.  This structure is used for recording all data types.

Details on recording specific data types is given below.

Recording energies and group-group energies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example

.. code-block:: python

    ljcut = FixLJCut(state, 'ljcut')
    ewald = FixChargeEwald(state, 'chargeEwald', 'all')
    def giveNextTurn(currentTurn):
        if currentTurn==0:
            return 1
        else:
            return currentTurn*2

    #this will record the potential energy of all atoms every 100 turns
    engDataSimple = state.dataManager.recordEnergy(handle='all', mode='scalar', interval=100)

    #this will record per-particle energies (as denoted by the ``mode``) 
    #every turn as given by the ``giveNextTurn`` function.  It will only 
    #record contributions due to ``ljcut`` and ``ewald`` fixes.  Any fix 
    #can be given in the ``fixes`` list.  
    engDataLogSpacing = state.dataManager.recordEnergy(handle='all', mode='vector', collectGenerator=giveNextTurn, fixes=[ljcut, ewald])
    
    #computes LJ interactions between ``groupA`` and ``groupB``
    engDataGroupGroup = state.dataManager.recordEnergy(handle='groupA', handleB='groupB', mode='scalar', interval=100, fixes=[ljcut])

    verlet = IntergatorVerlet(state)

    verlet.run(10000)

    #prints list of per-particle energy lists
    print engDataLogSpacing.vals 

**Arguments**

``handle``: Group handle for which energies will be compted.  Defaults to ``'all'``.

``mode``: ``'scalar'`` or ``'vector'``.  ``'scalar'`` computes the sum of relevant energies while ``'vector'`` computes per-particle energies represented as a python list.  Defaults to ``'scalar'``

``interval``: How often data is recorded.  Either ``interval`` or ``collectGenerator`` must be specified

``collectGenerator``: Function which takes the current turn on which data is being recorded and returns the next turn on which it should be recorded.  Either ``interval`` or ``collectGenerator`` must be specified

``fixes``: Fixes for which energy will be recorded.  Defaults to all active fixes

``handleB``: For group-group energies, the handle of the other group.  This is the other parameter which must be specified to perform group-group energy calculations.
    

Recording temperatures and kinetic energies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    ljcut = FixLJCut(state, 'ljcut')
    ewald = FixChargeEwald(state, 'chargeEwald', 'all')

    #this will record the temperature of all atoms every 100 turns
    tempDataScalar = state.dataManager.recordTemperature(handle='all', mode='scalar', interval=100)

    #this will record per-particle kinetic energies (as denoted by the ``mode``) 
    #every turn as given by the ``giveNextTurn`` function.  
    tempDataVector = state.dataManager.recordEnergy(handle='all', mode='vector', interval=100)
    
    verlet = IntergatorVerlet(state)

    verlet.run(10000)

    #prints list of tenoeratures followed by the turns at which those 
    #data points were recorded
    print tempDataScalar.vals, tempDataScalar.turns

**Arguments**

``handle``: Group handle for which temperature will be compted.  Defaults to ``'all'``.

``mode``: ``'scalar'`` or ``'vector'``.  ``'scalar'`` computes the temperature of the group given by ``handle`` while  ``'vector'`` computes per-particle kinetic energies represented as a python list.  Defaults to ``'scalar'``

``interval``: How often data is recorded.  Either ``interval`` or ``collectGenerator`` must be specified

``collectGenerator``: Function which takes the current turn on which data is being recorded and returns the next turn on which it should be recorded.  Either ``interval`` or ``collectGenerator`` must be specified



Recording pressures and virial coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    #this will record the system's pressure every 100 turns
    pressureData = state.dataManager.recordPressure(handle='all', mode='scalar', interval=100)

    #records pressure tensor
    pressureDataTensor = state.dataManager.recordPressure(handle='all', mode='tensor', interval=100)

    verlet = IntergatorVerlet(state)

    verlet.run(10000)

    #prints list of pressures
    print pressureData.vals 

**Arguments**

``handle``: Group handle for which temperature will be compted.  Defaults to ``'all'``.

``mode``: ``'scalar'`` or ``'tensor'``.  ``'scalar'`` computes the pressure of the group given by ``handle`` while  ``'tensor'`` computes pressuretensor due to that group.  Defaults to ``'scalar'``

``interval``: How often data is recorded.  Either ``interval`` or ``collectGenerator`` must be specified

``collectGenerator``: Function which takes the current turn on which data is being recorded and returns the next turn on which it should be recorded.  Either ``interval`` or ``collectGenerator`` must be specified

Recording volume and boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system bounding box can also be recorded.  From this volume, side lens, rates of volume change, etc, can easily be computed.

.. code-block:: python

    #this will record the system's pressure every 100 turns
    boundsData = state.dataManager.recordBounds(interval=100)

    verlet = IntergatorVerlet(state)

    verlet.run(10000)

    #prints list of pressures
    volumes = []
    xSideLengths = []
    for bounds in boundsData.vals:
        volumes.append(bounds.volume())
        xSideLengths.append(bounds.hi[0] - bounds.lo[0])

    #all the volumes computed
    print volumes 

**Arguments**

``interval``: How often data is recorded.  Either ``interval`` or ``collectGenerator`` must be specified

``collectGenerator``: Function which takes the current turn on which data is being recorded and returns the next turn on which it should be recorded.  Either ``interval`` or ``collectGenerator`` must be specified

Turning off recording
^^^^^^^^^^^^^^^^^^^^^

Recording of a data set can be stopped at any time by calling the ``stopRecord`` method of the ``DataManager`` 

.. code-block:: python
    
    myDataSet = state.dataManager.recordTemperature(handle='all', mode='scalar', interval=100)

    #turns off recording
    state.dataManager.stopRecord(myDataSet)
