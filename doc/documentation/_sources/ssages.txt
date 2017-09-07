Coupling with SSAGES
====================

DASH was contructed to efficiently couple to the advanced sampling engine SSAGES.  This allows for advanced sampling methods such as Basis Function sampling, Adaptive Bias Force sampling, Green's function sampling, umbrella sampling, and more to be performed at GPU speeds with minimal overhead.  

First, SSAGES must be compiled with DASH.  

.. code-block:: bash

    cd path/to/ssages/directory
    mkdir build
    cd build
    cmake .. -DDASH_SRC=/path/to/dash
    make

We also need to compile dash on its own in order to generate the python library 


Once both DASH and SSAGES are compiled, ensure that ``libSim.so`` is in your LD_LIBRARY_PATH environment variable, and you're ready to go.

SSAGES looks for two functions within your DASH python script - a ``setupSimulation`` function which initializes the :doc:`simulation state</state>` and a ``runSimulation`` function which takes as an argument a number of timesteps, and must run the simulation for that long.

Example
^^^^^^^

.. code-block:: python
    
    #import DASH as normal

    def setupSimulation():
        state = State()
        #create fixes, add atoms, prepare simulation state
        

        #function must return the simulation state
        return state

    #SSAGES calls this function using the state you returned in setupSimulation 
    #SSAGES has internally appended a special fix which applies 
    #operations required for you advanced sampling technique
    def runSimulation(state, numTurns):
        integ = IntegratorVerlet(state)
        nvt = FixNoseHoover(state, handle='myThermo', groupHandle='all', temp=300, timeConstant=100)
        state.activateFix(nvt)
        integ.run(numTurns)
        state.deactivateFix(nvt)


