Berendsen Barostat
==================

Overview
^^^^^^^^
Implements the Berendsen barostat for maintaining pressure at a specified set point pressure ``P`` through rescaling of the volume every ``applyEvery`` turns. 

Constructor
^^^^^^^^^^^
.. code-block:: python
    
   FixPressureBerendsen(state,handle,pressure,period,applyEvery)

Arguments

``state``
    Simulation state to which this fix is applied.

``handle``
    A name for this fix.  String type.

``groupHandle``
    The group of atoms to which this fix is applied.  String type.

``pressure``
    The set point pressure for the simulation.  Double type.

``period``
    The time constant associated with the barostat.  Double type.

``applyEvery``
    The number of turns to elapse between applications of this fix.  Integer type.


Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    setParameters(maxDilation)

Arguments

``maxDilation``
    The maximum factor by which the system volume may be scaled in any given turn.  Defaults to 0.00001.

Examples
^^^^^^^^

.. code-block:: python

    # create a simulation state to which we will apply the fix
    state = State()

    # make an instance of the fix; specify pressure of 0.5, period of 10, applyEvery 1
    fixPressure = FixPressureBerendsen(state,"npt",0.5,10,1)

    # call set parameters to change maxDilation to 0.0001
    fixPressure.setParameters(0.0001)

    # activate the fix
    state.activateFix(fixPressure)


