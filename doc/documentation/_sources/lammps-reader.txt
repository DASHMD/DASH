LAMMPS reader
=============

Overview
^^^^^^^^

The LAMMPS reader allows LAMMPS configurations to be read into the :doc:`simulation state</state>`.  The LAMMPS reader parses input and data files and adds the configuration to the current lists of atoms, bonds, angles, dihedrals, and imporpers.  Items such as thermostats, barostats, and data recorders are not read.

One common simulation workflow is to read in multiple lammps configurations, one for each molecule type in the system, and then populate the system using the :doc:`molecule</molecule>` object.

The LAMMPS reader works by parsing the supplied LAMMPS input and data files and adding items it reads into the simulation state through the standard python interface.  As such, it can be modified or extended to accommodate any unsupported features of LAMMPS, or even other simulation engines.

Examples
^^^^^^^^

    
.. code-block:: python
    
   sys.path.append('simulation_directory/src/util_py')
   from LAMMPS_Reader import LAMMP_Reader

   #declare fixes which the lammps reader will populate
   nonbond = FixLJCut(state, 'ljcut')
   bondHarmonic = FixBondHarmonic(state, 'harmonic')

   reader = LAMMPS_reader(state=state, atomTypePrefix='...', setBounds=True/False, 
        nonbondFix=nonbond, bondFix=bondHarmonic, angleFix=..., dihedralFix=...,
        improperFix=...)
   
   #now the reader is set up and we can read in files
   reader.read(inputFns=['input1.in', 'input2.in'], dataFn='data.dat')

   #can read in more files
   reader.read(inputFns=['moreInput.in'], dataFn=moreData.dat)





