Molecules
=========

Overview
^^^^^^^^

:doc:`Molecules</molecule>` are a tool for duplicating and manipulating groups of atoms.  Molecules are particularly useful for initializing systems. 

.. code-block:: python

    #assigns atoms to a molecule
    molec = state.createMolecule(ids=[...])
    #create a complete copy of the molecule including bonds, angles, etc.
    duplicate = state.duplicateMolecule(molec)
    #move the molecule by x=10
    duplicate.translate(Vector(10, 0, 0))
    #rotate the molecule by pi radians around the axis (1, 0, 0)
    duplicate.rotate(Vector(1, 0, 0), pi)

Creating a molecule groups already-existing atoms into a molecule.  This molecule can then be duplicated, translated, and rotated.  Molecules can be accessed through the ``state.molecules`` member, which is a python list.  

.. code-block:: python

    state.deleteMolecule(molec)

Molecules can also be deleted. Deleting a molecule deletes the member atoms and all associated bonds, angles, etc.
