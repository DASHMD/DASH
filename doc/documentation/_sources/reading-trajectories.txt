Reading trajectories
====================

Overview
^^^^^^^^

Trajectories in xml format can be read in as follows:

First load the file.  Every ``State`` object has a ``readConfig`` member.


.. code-block:: python

    state.readConfig.loadFile('myRestart.xml')

A file can store one or many configurations.  To iterate to the first configuration

.. code-block:: python

    state.readConfig.next()

Or to iterate to the last configuration

.. code-block:: python

    state.readConfig.prev()

To move by a certain number of trajectories

.. code-block:: python

    #advance by 3 snapshots within this xml file
    state.readConfig.moveBy(3)

    #move backwards by 3 snapshots within this xml file
    state.readConfig.moveBy(-3)

Calling these commands again will iterate forwards or backwards over the set of trajectories.  ``next`` and ``prev`` methods will return ``True`` if a valid configuration has been read or ``False`` if you are at the end of the series of trajectories.

It is important that you initialize fixes **after** the configuration has been read such that bonds, angles, etc, are property read in.  This restriction will be removed in future releases.


**Other ways to read trajectories**

One can also read in LAMMPS trajectories using the :doc:`LAMMPS reader</lammps-reader>`, or manually assign atom configurations using the python interface.
