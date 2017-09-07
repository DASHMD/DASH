Ewald potential with long-range Coulomb interactions
====================================================

Overview
^^^^^^^^

Long range Coulomb interactions implemented from Deserno and Holm, J.Chem.Phys. 109, 7678, 1998. The idea of the method is to accurately include periodic boundary conditions in Coulomb potential calculation,and allow particles to interact with infinite amount of simulation box images. This is done by separation of Coulomb potential into short-range pairwise part and long-range part. Short-range part force is:

.. math::
   F_{\rm short}({\bf r}_{ij}) =  \left[\begin{array}{cc}  q_i q_j \left(\frac{2\alpha}{\sqrt{\pi}}{\rm exp}(-\alpha^2 r_{ij}^2)+\frac{{\rm erfc}(\alpha r)}{r}\right)\frac{{\bf r}_{ij}}{r_{ij}^2},& r<r_{\rm cut}\\
                    0, & r\geq r_{\rm cut}
                    \end{array}\right.
   
The long-range part calculation requires mapping pariticle charges to a mesh, calculating energy and electric field  at mesh points assuming large number of periodic images through Fourier space. Then forces on particles are calculated through:

.. math::
   F_{{\rm long},i} = q_i \sum_{{\bf r}_{\rm p}\in {\mathcal M}} {\bf E}({\bf r}_{\rm p}) W({\bf r}_{\rm i}-{\bf r}_{\rm p})


where :math:`q_i, q_j` are charges of particles :math:`i,j`, :math:`r_{ij}` is the distance between particles :math:`i,j`, :math:`\alpha` is splitting parameter, :math:`{\bf r}_{\rm p}` is coordinates of mesh points, :math:`W({\bf r})` is charge assignment function and :math:`r_{\rm cut}` is cutoff distance. 

``FixChargeEwald`` reports root mean square (RMS) force error from analytical approximation.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
Adding Fix 

.. code-block:: python

    FixChargeEwald(state=..., handle=...,group_handle=...)

Arguments 

``state``
   state object to add the fix.

``handle``
  A name for the fix. 
  
``group_handle``
  Group name to apply potential. 

Setting parameters from within the Python environment is done with ``setParameters``. 

.. code-block:: python

    setParameters(szx=...,szy=...,szz=..., r_cut=..., interpolation_order=...)
    setParameters(sz=..., r_cut=..., interpolation_order=...)

Arguments 

``szx,szy,szz``
    number of mesh points in x,y,z axis. Can be set to 32,64,128,256,512,1024
    
``sz``
    number of mesh points for all axes. Can be set to 32,64,128,256,512,1024
    
``r_cut``
    cutoff raduis for a short-range pairwise part. By default value is taken from ``state``.

``interpolation_order``
    number of mesh points included into charge assignment function. Implemented orders are 1,3. Default is 3.

It is possible to set required RMS error instead of mesh size with ``setError``

.. code-block:: python

    setError(error=..., r_cut=..., interpolation_order=...)

Arguments 

``error``
    required root mean square (RMS) force error.
    
``r_cut``
    cutoff raduis for a short-range pairwise part. By default value is taken from ``state``.

``interpolation_order``
    number of mesh points included into charge assignment function. Implemented orders are 1,3. 
    
It is possible to avoid updating long-range part every timestep with ``setLongRangeInterval``

.. code-block:: python

    setLongRangeInterval(interval=...)
    
Arguments 

``interval``
    number of timestep between long-range part updates. By default ``FixChargeEwald`` calculates long-range part every timestep.

Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #adding charge fix
    charge=FixChargeEwald(state, "charge", "all")
    
Setting parameters in python

.. code-block:: python

    #64 grid points in each dimension, cutoff of rCut-1
    #interpolating charge to three mesh points
    charge.setParameters(64, state.rCut-1, 3);

    #alternatively, one could let DASH determinine the 
    #grid discretization by setting an error tolerance 
    #(1e-2 here)
    #charge.setError(1e-2)


Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(charge)



