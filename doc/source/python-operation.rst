Python Operation
================

Overview
^^^^^^^^

The Python Operation class injects a python function into the simulation runtime to be called every ``operateEvery`` turns.  Python Operations are performed asynchronously, meaning that the simulation continues to run while the operation is being performed.  As a result, also arbitrarily complex function can be computed in python with little performance impact.  

The low overhead of Python Operations allows for data to be flexibly computed routinely during production runs.  

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    storedValues = [] 
    def computeDist(currentTurn):
        #compute distance between atoms 0 and 3.  
        storedValues.append(state.atoms[3].pos - state.atoms[0].pos)
        
    myOperation = PythonOperation(handle='myOp', operateEvery=50, operation=computeDist)
    #now computeDist will be called every 50 turns. 
    #it can access state as it would between runs - all the data is on the CPU
    state.activatePythonOperation(myOperation)

    #run the simulation
    integrator.run(10000)


    #turn off python operation
    state.deactivatePythonOperation(myOperation)
