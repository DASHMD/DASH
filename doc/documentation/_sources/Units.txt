Units
==============

Overview
^^^^^^^^

Simulations in Dash can use either real or reduced/Lennard-Jones units.  This is specified by updating the ``units`` attribute of the ``state``.  Lennard-Jones units are the default, so it is imporant to select the desired units before setting simulation parameters.


Setting Real Units
^^^^^^^^^^^^^^^^^^

Dash can be set to real (or LJ) units by invoking the ``setReal()`` (or ``setLJ()``) method of the ``units`` property in the ``state``.  Real units are the following:

`Energy`: kcal/mol

`Distance`: Angstroms

`Time`: femptoseconds

`Temperature`: Kelvin

`Mass`: atomic mass units (amu)

`Charge`: electronic charge

`Pressure`: atmospheres


**Example**

The following code sets the units of a simulation to real units and then back to reduced units.

Setting the units also sets the timestep to default values of 1.0 fs for real units and 0.005 for LJ units.

.. code-block:: python
	
	#set real units
	state.units.setReal()
	
	#return to LJ/reduced units
	state.units.setLJ()


