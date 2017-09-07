Intergrator Verlet
====================

Overview
^^^^^^^^

Integrating state via two step velocity-Verlet

.. math::
 {\bf v}_i\left(t+\frac{1}{2}\Delta t\right) &=& {\bf v}_i\left(t\right) + \frac{1}{2}\frac{{\bf f}_i\left(t\right)}{m_i}\Delta t\\
 {\bf r}_i\left(t+\Delta t\right) &=& {\bf r}_i\left(t\right) + {\bf v}_i\left(t+\frac{1}{2}\Delta t\right)\Delta t\\
 {\bf v}_i\left(t+\Delta t\right) &=& {\bf v}_i\left(t+{\Delta t}/{2}\right)+\frac{1}{2}\frac{{\bf f}_i\left(t+\Delta t\right)}{m_i}\Delta t


where :math:`{\bf r}_{i}` is coordinate of particle :math:`i`, :math:`{\bf v}_i` is  velocity of particle :math:`i`, :math:`{\bf f}_i` is sum of all forces acting on particle :math:`i`, :math:`m_i` is  mass particle :math:`i`, and :math:`\Delta t` is timestep size.
The timestep size is set through ``state`` variable:

.. code-block:: python

    state.dt=0.005


Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

Constructor

.. code-block:: python

    IntegratorVerlet(state=...)

Arguments 

``state``
   state object.

Integrating state is done with ``run``. 

.. code-block:: python

    run(numTurns=...)

Arguments 

``numTurns``
    number of timestep to make.
   
    
TODO Write Output?


Examples
^^^^^^^^
Adding the integrator 

.. code-block:: python

    integrater = IntegratorVerlet(state)

    
Setting parameters in python

.. code-block:: python

    state.shoutEvery=1000
    state.dt=0.005

integrating system forward in time

.. code-block:: python

    #run 1E5 timesteps
    integrater.run(100000)


