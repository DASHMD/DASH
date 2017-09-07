Nose-Hoover Thermostat and Barostat
===================================

Overview
^^^^^^^^
Implements Nose-Hoover dynamics using the equations of motion as outlined in Tuckerman et. al, J. Phys. A: Math. Gen 39 (2006) 5629-5651.  Allows for dynamics from the NVT or NPT ensembles to be simulated.  The total Liouville operator implemented for NPT dynamics is given by (from Tuckerman et. al 2006, p.5641):

.. math:: 

    iL = iL_1 + iL_2 + iL_{\epsilon,1} + iL_{\epsilon,2} + iL_{T-baro} + iL_{T-Part}

    iL_1 = \sum\limits_{i=1}^N \bigl[\frac{\mathbf{p}_i}{m_i} + \frac{p_{\epsilon}}{W} \mathbf{r}_i \bigl] \cdot \frac{\partial}{\partial \mathbf{r}_i}

    iL_2 = \sum\limits_{i=1}^N \bigl[\mathbf{F}_i - \alpha \frac{p_{\epsilon}}{W}\mathbf{p}_i \bigl] \cdot \frac{\partial}{\partial \mathbf{p}_i}

    iL_{\epsilon,1} = \frac{p_{\epsilon}}{W} \frac{\partial}{\partial \epsilon}

    iL_{\epsilon,2} = G_{\epsilon} \frac{\partial}{\partial p_{\epsilon}}

    \text{where } G_{\epsilon} = \alpha \sum\limits_i \frac{\mathbf{p}_i^2}{m_i} + 
    \sum\limits_{i=1}^N \mathbf{r}_i \cdot \mathbf{F}_i - 3 V \frac{\partial U}{\partial V} - PV
   

Here, :math:`\mathbf{p}_i` and :math:`\mathbf{r}_i` are the particle momenta and positions, :math:`\mathbf{F}_i` are the forces on the particles, :math:`p_{\epsilon}` and :math:`W` are the barostat momenta and masses, :math:`\alpha` is a factor of :math:`1+\frac{1}{N}`, and :math:`P` and :math:`V` are the set point pressure and instantaneous volume, respectively.


The corresponding propagator for the NPT ensemble is then given by:

.. math:: 

    \exp(iL \Delta t) = \exp (iL_{T-baro} \frac{\Delta t}{2}) \exp (iL_{T-part} \frac{\Delta t}{2}) \exp (iL_{\epsilon,2} \frac{\Delta t}{2}) \\
    \times \exp (iL_2 \frac{\Delta t}{2}) \exp (iL_{\epsilon,1} \Delta t) \exp(iL_1 \Delta t) \exp(iL_2 \frac{\Delta t}{2}) \\
    \times \exp(iL_{\epsilon,2} \frac{\Delta t}{2}) \exp(iL_{T-part} \frac{\Delta t}{2}) \exp(iL_{T-baro} \frac{\Delta t}{2})

The barostat variables and particles are separately thermostatted; in each case, a chain of 3 thermostats is used.  Integration is accomplished via a first order Suzuki-Yoshida integration scheme.  

Constructor
^^^^^^^^^^^
.. code-block:: python

    FixNoseHoover(state,handle,groupHandle)


Arguments

``state``
    The simulation state to which the fix is applied.

``handle``
    The name of the fix.  String type.

``groupHandle``
    The group of atoms to which this fix is to be applied.  String type.


Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
The Nose-Hoover Barostat/Thermostat set points are set via the Python member functions.  

Temperature set points may be input through any of the following commands:

.. code-block:: python

   setTemperature(temp,timeConstant)
   setTemperature(temps,intervals,timeConstant)
   setTemperature(tempFunc, timeConstant)

The first method holds the temperature of ``groupHandle`` at a set point of ``temp``.  The second linearly interpolates between the temperatures given in the list ``temps`` at the fractions through the current run given by ``intervals``.  The third allows the set point to be determined at each turn via ``tempFunc``.  The same scheme is used to set pressure.

Arguments: 

``temp``
    The temperature set point for the simulation.  Float type.

``timeConstant``
    The time constant associated with the thermostat variables.  Float type.

``temps``
    A list of temperature set points to be used throughout the simulation.  List of floats.
 
``intervals``
    A list of fractions through the next run which correspond to the temperatures given in ``temps``. List of floats.
 
``tempFunc``
    The temperature set point, implemented as a python function. 
 
    
    
Likewise, specification of a set point pressure may be accomplished through any of the following commands:

.. code-block:: python

   setPressure(mode,pressure,timeConstant)  
   setPressure(mode,pressures,intervals,timeConstant)
   setPressure(mode,pressFunc,timeConstant)

Arguments:

``mode``
    The mode in which cell deformations occur; options are "ISO" or "ANISO".  With mode "ISO", the internal stress tensor is averaged across the three normal components (or 2, for 2D simulations), and a uniform scale factor for the dimensions emerges.  For "ANISO", the components of the internal stress tensor are not averaged and the individual dimensions are scaled independently.

``pressure``
    The set point pressure for the simulation.  Float type.

``timeConstant``
    The time constant associated with the barostat variables.  Float type.

``pressures``
    A list of pressure set points to be used through the simulation.  List of floats.
 
``intervals``
    A list of fractions through the next run which correspond to the pressures given in ``pressures``. List of floats.
 
``pressFunc``
    The pressure set point, implemented as a python function.


For NPT dynamics, both ``setTemperature`` and ``setPressure`` should be called.  They can be called in any order before the simulation is run.



Examples
^^^^^^^^

Example 1: Nose-Hoover Thermostat (NVT Ensemble) - constant set point temperature

.. code-block:: python
    
    # create a simulation state
    state = State()

    # make an instance of the fix
    fixNVT = FixNoseHoover(state, "nvt", "all")

    # assign a set point temperature of 300K with time constant 100*state.dt
    fixNVT.setTemperature(300.0, 100*state.dt)

    # activate the fix
    state.activateFix(fixNVT)


Example 2: Nose-Hoover Barostat & Thermostat (NPT Ensemble) - constant set point temperature & pressure

.. code-block:: python

    # create a simulation state
    state = State()

    # make an instance of the fix
    fixNPT = FixNoseHoover(state, "npt", "all")

    # assign a set point temperature and time constant 100*state.dt
    fixNPT.setTemperature(250.0, 100*state.dt)

    # assign a set point pressure and time constant 1000*state.dt with isotropic cell deformations
    fixNPT.setPressure("ISO", 1.0, 1000*state.dt)

    # activate the fix
    state.activateFix(fixNPT)


Example 3: Setting temperature via ``temperature`` and ``intervals``

.. code-block:: python
    
    # create a simulation state
    state = State()

    # make an instance of the fix
    fixNVT = FixNoseHoover(state, "nvt", "all")

    # assign a set point temperature of 300K with time constant 100*state.dt
    fixNVT.setTemperature(temps=[100, 500, 400], intervals=[0, 0.2, 1.0], 100*state.dt)

    # activate the fix
    state.activateFix(fixNVT)

    integrator = IntegratorVerlet(state)

    #Will sweep from temp=100 to temp=500 between turns 0 and 2000, then 500 and 400 between turns 2000 and 10000
    integrator.run(10000)


Example 4: Setting temperature via ``tempFunc``

.. code-block:: python
    
    def randomTemp(turnRunBegan, turnRunEnds, currentTurn):
        return 100 + 50*random() * (currentTurn-turnRunBegan)/(turnRunEnds-turnRunBegan)

    # create a simulation state
    state = State()

    # make an instance of the fix
    fixNVT = FixNoseHoover(state, "nvt", "all")

    # assign a set point temperature of 300K with time constant 100*state.dt
    fixNVT.setTemperature(tempFunc=randomTemp, 100*state.dt)

    # activate the fix
    state.activateFix(fixNVT)

    integrator = IntegratorVerlet(state)

    #Will use the value returned by randomTemp each turn as the setpoint
    integrator.run(10000)


