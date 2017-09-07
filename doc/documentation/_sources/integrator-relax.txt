Intergrator FIRE relaxation
===========================

Overview
^^^^^^^^

Integrating state via FIRE algorithm -- Fast Internal Relaxation Engine
see (Erik Bitzek et al.  PRL 97, 170201, (2006))

FIRE algorithm:

1. MD: calculate :math:`{\bf r}`,  :math:`{\bf f}=-\nabla E(\bf r)`, and :math:`{\bf v}` using any common MD integrator; check for convergence.
2. F1: calculate :math:`P = F \cdot v`.
3. F2: set :math:`{\bf v} \rightarrow (1-\alpha) {\bf v}+ \alpha \hat{\bf f} \left| {\bf v}\right|`.
4. F3: if :math:`P > 0` and the number of steps since :math:`P` was negative is larger than :math:`N_{\rm min}`, increase the time step :math:`\Delta t \rightarrow min\left(\Delta t \,f_{\rm inc}; \Delta t_{\rm max}\right)` and decrease :math:`\alpha\rightarrow  \alpha\, f_\alpha`. 
5. F4: if :math:`P \leq 0`, decrease time step :math:`\Delta t \rightarrow \Delta t\,f_{\rm dec}`, freeze the system :math:`{\bf v} = 0` and set :math:`\alpha` back to :math:`\alpha_{\rm start}` .
6. F5: return to MD.


where :math:`{\bf r}` is vector of coordinates, :math:`{\bf v}` is vector of velocities, :math:`{\bf f}_i`  is vector of forces, and :math:`\Delta t` is timestep size.
Algorithm parameters are: :math:`\alpha_{\rm start}, f_\alpha, f_{\rm inc}, f_{\rm dec}, N_{\rm min}, \Delta t_{\rm max}`



Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Constructor

.. code-block:: python

    IntegratorRelax(state=...)

Arguments 

``state``
   state object.

Setting parameters from within the Python environment is done with ``set_params``. 
   
.. code-block:: python

    set_params(alphaInit=..., alphaShrink=..., dtGrow=..., dtShrink=..., delay=..., dtMax_mult=...)
    

Arguments 

``alphaInit``
    initial value of :math:`\alpha`. Corresponds to :math:`\alpha_{\rm start}` in algorithm description.

``alphaShrink``
    corresponds to :math:`f_\alpha` in algorithm description.
    
``dtGrow``
    corresponds to :math:`f_{\rm inc}` in algorithm description.

``dtShrink``
    corresponds to :math:`f_{\rm dec}` in algorithm description.
    
``delay``
    corresponds to :math:`N_{\rm min}` in algorithm description.
    
``dtMax_mult``
    maximum value of :math:`\Delta t` relative to ``state.dt``. Corresponds to :math:`\Delta t_{\rm max}/\Delta t_{\rm initial}` in algorithm description.

Integrating state is done with ``run``. 

Default values:

    alphaInit = 0.1

    alphaShrink = 0.99

    dtGrow = 1.1

    dtShrink = 0.5

    delay = 5

    dtMax_mult = 10

.. code-block:: python

    run(numTurns=...,fTol=...)

Arguments 

``numTurns``
    number of timestep to make.
   
``fTol``
    required force tolerance. When :math:`\left|{\bf f}\right|<fTol`, ``run`` returns.


Examples
^^^^^^^^
Adding the integrator 

.. code-block:: python

    integrater = IntegratorRelax(state)

    
Setting parameters in python

.. code-block:: python

    state.shoutEvery=1000
    state.dt=0.005


Relax the system

.. code-block:: python

    #run 1E5 timesteps
    integrater.run(100000)


