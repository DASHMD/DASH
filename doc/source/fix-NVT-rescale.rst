Isokinetic Thermostat
===================================

Overview
^^^^^^^^
The FixNVTRescale permits rescaling of the velocities every ``applyEvery`` turn in the simulation for maintaining the temperature at some specified set point ``T``.  Temperature set points may be specified as a constant float value, as a list of temperatures with associated time intervals, or as a python function.  


Constructor
^^^^^^^^^^^

An isokinetic thermostat may be instantiated using any of the following constructors:

.. code-block:: python

    FixNVTRescale(state,handle,groupHandle,temp,applyEvery)
    FixNVTRescale(state,handle,groupHandle,tempFunc,applyEvery)
    FixNVTRescale(state,handle,groupHandle,intervals,temps,applyEvery)


Arguments

``state``
    The simulation State to which this fix is applied.

``handle``
    The name for this fix.  String type.

``groupHandle``
    The group of atoms to which this fix is applied.  String type.

``temp``
    The temperature to which the kinetic energy will be rescaled.  Float type.

``applyEvery``
    The number of turns to elapse between applications of this fix.  Integer type.

``tempFunc``
    The function specifying the set point temperatures throughout the simulation to which the kinetic energy will be rescaled.  Python function.

``intervals``
    A list of fractions through the current run for the corresponding list of temperature set points.  List of floats.

``temps``
    A list of temperature set points to be used throughout the simulation.  List of floats.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

This Fix contains no python member functions.


Examples
^^^^^^^^

.. code-block:: python

    # create a simulation state to which we will apply the fix
    state = State()

    # make an instance of the fix; T = 250.0 K, applyEvery = 7
    fixNVT = fixNVTRescale(state,"nvt","all",250.0,7)

    # activate the fix
    state.activateFix(fixNVT)

