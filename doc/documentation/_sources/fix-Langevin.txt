Fix Langevin
====================

Overview
^^^^^^^^

FixLangevin applies stochastic forces to particles in the system via Langevin dynamics

.. math::
  {\bf f}_i = -\gamma {\bf v}_i+\sqrt{6k_{\rm B}T\gamma/\Delta t }{\bf W}_i


where :math:`{\bf v}_i` is  velocity of particle :math:`i`, :math:`\Delta t` is timestep size, :math:`T` is temperature, :math:`\gamma` is a drag parameter, and :math:`{\bf W}_i` is a Wiener vector.
  
  
Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Constructors

.. code-block:: python

    FixLangevin(state=..., handle=...,  groupHandle=...,  temp=...)
    FixLangevin(state=..., handle=...,  groupHandle=...,  tempFunc=...)
    FixLangevin(state=..., handle=...,  groupHandle=...,  intervals=... , temps=...)

Arguments 

``state``
   state object.

``handle``
  A name for the fix. 
  
``group_handle``
  Group name to apply fix. 
  
``temp``
   Constant Temperature. 

There are few options to change temperature dynamically:
 
``tempFunc``
   Python function that returns a temperature at timestep.
   
``intervals, temps``
   Python lists of temperatures and intevals for linear interpolation.

   
Setting parameters from within the Python environment is done with ``setParameters``. 
   
.. code-block:: python

    setParameters(seed=..., gamma=...)

Arguments 

``seed``
    seed for random number generator
    
``gamma``
     :math:`\gamma` drag parameter.
     
     
Examples
^^^^^^^^
Adding the integrator 

.. code-block:: python

    fixLangevin=FixLangevin(state, handle='Lang',groupHandle='all',temp=2.0)

    
Setting parameters in python

.. code-block:: python

    fixLangevin.setParameters(seed=1234,gamma=1.0)

Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(fixLangevin)


