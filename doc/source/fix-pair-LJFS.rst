Force shifted LJ Pair potential
===============================

Overview
^^^^^^^^

Define a pair potential energy function corresponding to the modified Lennard-Jones potential

.. math::
   V(r_{ij}) =  \left[\begin{array}{cc} 4 \varepsilon \left( \left(\frac{\sigma}{r_{ij}}\right)^{12} -
                    \left(\frac{\sigma}{r_{ij}}\right)^{6}\right)-F_{\rm LJ}(r_{\rm cut})r_{ij},& r<r_{\rm cut}\\
                    0, & r\geq r_{\rm cut}
                    \end{array}\right.

where :math:`r_{ij}` is the distance between particles :math:`i,j` :math:`\varepsilon, \sigma` are Lennard-Jones potential parameters, and that must be set to define the interaction between atoms, :math:`r_{\rm cut}` is cutoff distance, and :math:`F_{\rm LJ}(r_{ij})` is the original Lennard-Jones force:
                
.. math::
    F_{\rm LJ}(r) =\frac{\partial V_{\rm LJ}(r)}{\partial r} (r).
    
This modified Lennard-Jones potential has force equal to 0 at cutoff distance.

parameters of potential can be defined directly within the python input script, read from a LAMMPS data file (using the LAMMPS reader utility), read from a NAMD input file (using the NAMD reader utility) or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
Adding Fix 

.. code-block:: python

    FixLJCutFS(state=..., handle=...)

Arguments 

``state``
   state object to add the fix

``handle``
  A name for the fix. 


Setting parameters from within the Python environment is done with ``setParameter``. 

.. code-block:: python

    setParameter(param=...,handleA=...,handleB=...,val=...,)

Arguments 

``param``
    name of parameter to set. Can be ``eps``, ``sig``, ``rCut``.
    
    ``rCut`` parameter has a default value equal to ``state.rCut``. All other parameters have to be set manually.
    
``handleA``, ``handleB``
    a pair of type names to set parameters 

``val``
    value of the parameter.



It is also possible to get value of the parameters within the Python environment with ``getParameter``. 

.. code-block:: python

    val = getParameter(param=...,handleA=...,handleB=...)

Arguments 

``param``
    name of parameter to set. Can be ``eps``, ``sig``, ``rCut``

``handleA``, ``handleB``
    pair of type names to set parameters for

``val``
    value of the  parameter



Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #adding force-shifted Lennard-Jones potential
    ljcut = FixLJCutFS(state, handle='ljcut')
    
Setting parameters in python

.. code-block:: python

    ljcut.setParameter(param='eps', handleA='A', handleB='B', val=1.0)
    ljcut.setParameter(param='sig', handleA='B', handleB='B', val=1.0)

Setting same parameters for all types in python

.. code-block:: python

    ljsig=1.0
    ljeps=1.0
    #list of all types
    types=['A','B','C','S','P','N']
    for A in types:
        for B in types:
            ljcut.setParameter(param='eps', handleA=A, handleB=B, val=ljeps)
            ljcut.setParameter(param='sig', handleA=A, handleB=B, val=ljsig)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(ljcut)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ``rCut``, ``eps``, ``sig``

