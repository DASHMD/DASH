Bounds
==============

Overview
^^^^^^^^

``Bounds`` objects are used to define volumes in space.  The ``state`` must have a ``bounds`` property specifying the simulation volume; smaller sections of this total volume can be used to perform various functions, such as populating certain regions in space with atoms.


Creating and Modifying Bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class Bounds(** state, lo, high **)**

``state``: The simulation state.

``lo``: A Vector object indicating the lower bounds in the x,y, and z directions.

``high``: A Vector object indicating the upper bounds in the x,y, and z directions.

**Attributes**

The following attributes and methods of the ``Bounds`` object are available:

``lo``: A Vector object indicating the lower bounds in the x,y, and z directions (read/write).

``high``: A Vector object indicating the upper bounds in the x,y, and z directions (read/write).

``volume``: The volume of the space enclosed by the ``Bounds`` (read-only).

Methods
"""""""

**bounds.copy()**

This method returns a copy of the ``Bounds`` object on which it is invoked.


**bounds.vectorInBounds(** Vector v **)**

This method returns ``True`` if the argument Vector object ``v`` describes a position within the bounds.

**bounds.atomInBounds(** Atom a **)**

This method returns ``True`` if the position of atom ``a`` is within the bounds.

**bounds.minImage(** Vector v **)**

This method returns a Vector containing the periodic image of the position described by the input Vector ``v``.

**Example**

The following example illustrates the syntax used to initiaize a simulation box.

.. code-block:: python
	
	#set the bounds for a 5x5x5 box
	state.bounds = Bounds(state, lo=Vector(0, 0, 0), hi=Vector(5, 5, 5))
	
	#get the volume of our box
	volume = state.bounds.volume


