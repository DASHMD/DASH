TICG pair potential
====================

Overview
^^^^^^^^

Define a pair potential energy function corresponding to the soft core repulsive potential with energy proportional to volume of intersecting spheres


.. math::
   V(r_{ij}) =  \left[\begin{array}{cc} -C\frac{(r_{ij} - 2 R)^2 (r_{ij} + 4 R)}{16 R^3}, & r<r_{\rm cut}\\
                    0, & r\geq r_{\rm cut}
                    \end{array}\right.


where :math:`r_{ij}` is the distance between particles :math:`i,j` :math:`C` is potential strength, :math:`R` is the sphere radius. Note that cutoff distance is two times :math:`R`.

parameters of potential can be defined directly within the python input script or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
Adding Fix 

.. code-block:: python

    FixTICG(state=..., handle=...)

Arguments 

``state``
   state object to add the fix.

``handle``
  A name for the fix. 


Setting parameters from within the Python environment is done with ``setParameter``. 

.. code-block:: python

    setParameter(param=...,handleA=...,handleB=...,val=...,)

Arguments 

``param``
    name of parameter to set. Can be ``C``, ``rCut``.
    
    ``rCut`` parameter has a default value equal to ``state.rCut``. Note that ``R`` is half of ``rCut``.
    
``handleA``, ``handleB``
    a pair of type names to set parameters. 

``val``
    value of the parameter.



it is possible to get value of the parameters within the Python environment with ``getParameter``. 

.. code-block:: python

    val = getParameter(param=...,handleA=...,handleB=...)

Arguments 

``param``
    name of parameter to set. Can be ``C``, ``rCut``.

``handleA``, ``handleB``
    pair of type names to set parameters for.

``val``
    value of the  parameter.



Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #adding  TICG potential
    ticg = FixTICG(state, handle='TICG')
    
Setting parameters in python

.. code-block:: python

    ticg.setParameter(param='eps', handleA='A', handleB='B', val=1.0)
    ticg.setParameter(param='sig', handleA='B', handleB='B', val=1.0)

Setting same parameters for all types in python

.. code-block:: python

    #list of all types
    types=['A','B','C','S','P','N']
    for A in types:
    for B in types:
        ticg.setParameter(param='C', handleA=A, handleB=B, val=1.0)
        ticg.setParameter(param='rCut', handleA=A, handleB=B, val=0.2)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(ticg)

LAMMPS data file parameter order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ``rCut``, ``C``

