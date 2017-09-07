Writing Trajectories
====================

Overview
^^^^^^^^

Write a restart file every ``writeEvery`` turns in ``xyz`` or DASH-specific ``xml`` format.  A ``WriteConfig`` object must be created as shown below.  At this point, the ``write()`` method can be called to immediately write a configuration, or the ``WriteConfig`` can be activated and configurations will be written every ``writeEvery`` turns.  

Output is performed asynchonously, allowing restarts to be written frequently with minimal performance impact.

If units are set as real and ``format`` is ``xyz``, atomic numbers for the ``xyz`` for will be guessed from the atomic mass.  If the atomic number cannot be guessed, the atom type will be used.

Examples
^^^^^^^^
Basic usage

.. code-block:: python

    #Create WriteConfig object which will write configurations 
    #to myRestartFile.xml every 1000 turns.  Multiple configurations 
    #are written to the same xml file
    writeConfig = WriteConfig(state, fn="myRestartFile", writeEvery=1000, handle="writer1", format="xml")

    #write a configuration
    writeConfig.write() 

    #active the writer.  Now turns will be written every 100 turns.
    state.activateWriteConfig(state)
    
    
Writing one file per config (can be done with any format)

.. code-block:: python

    #Adding a * to the file name tells DASH to create 
    #one file per configuration where the current 
    #turn is substituted for *
    oneFilePerConfig = WriteConfig(state, fn="myRestartFile_*", writeEvery=1000, handle="writer1", format="xml")

Writing ``xyz`` files

.. code-block:: python

    #Writing xyz 
    oneFilePerConfig = WriteConfig(state, fn="myRestartFile_*", writeEvery=1000, handle="writer1", format="xyz")

Constructor
^^^^^^^^^^^

.. code-block:: python

    WriteConfig(state, fn, handle, format, writeEvery, groupHandle, unwrapMolecules) 

Arguments 

``state``
    State to output

``fn``
    Output filename.  Named argument.  File extension is automatically appended.

``handle``
    A name for the object.  Named argument.

``writeEvery``
    Write file every ``writeEvery`` turns.  Named argument.

``groupHandle``
    Group of atoms to output.  Named argument.  Defaults to ``all``.

``unwrapMolecules``
    Unwrap ``Molecule`` objects across periodic boundaries

