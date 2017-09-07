DSF Charge Pair potential
=========================

Overview
^^^^^^^^

Pairwise force shifted damped Coulomb as defined by
Christopher J. Fennel and J. Daniel Gezelter J. Chem. Phys (124), 234104 2006
Eqn 19.

.. math::
   F(r_{ij}) =  \left[\begin{array}{cc}  q_i q_j \left(\frac{{\rm erf}(\alpha r)}{r^2} +\frac{2\alpha}{\sqrt{\pi}} \frac{{\rm exp}(-\alpha^2 r^2    )}{r}  -\frac{{\rm erf}(\alpha r_{\rm cut})}{r_{\rm cut}^2}+\frac{2\alpha}{\sqrt{\pi}}\frac{{\rm exp}(-\alpha^2 r_{\rm cut}^2)}{r_{\rm cut}}\right),& r<r_{\rm cut}\\
                    0, & r\geq r_{\rm cut}
                    \end{array}\right.


where :math:`q_i, q_j` are charges of particles :math:`i,j`, :math:`r_{ij}` is the distance between particles :math:`i,j`, :math:`\alpha` is  convergence parameter, and :math:`r_{\rm cut}` is cutoff distance.


parameters of potential can be defined directly within the python input script, or read from a restart file.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
Adding Fix 

.. code-block:: python

    FixChargePairDSF(state=..., handle=...,group_handle=...)

Arguments 

``state``
   state object to add the fix.

``handle``
  A name for the fix. 
  
``group_handle``
  A group name to apply potential. 

Setting parameters from within the Python environment is done with ``setParameters``. 

.. code-block:: python

    setParameters(alpha=...,r_cut=...)

Arguments 

``alpha``
    name of parameter to set. Can be ``eps``, ``sig``, ``rCut``.
    
    ``rCut`` parameter has a default value equal to ``state.rCut``. 
    
``r_cut``
    Cutoff radius for charge calculations

Default parameter values are :math:`\alpha=0.25`, :math:`r_{\rm cut}=9.0`, assuming :math:`\sigma_{\rm LJ}=1.0` 


Examples
^^^^^^^^
Adding the fix

.. code-block:: python

    #adding charge fix
    charge=FixChargePairDSF(state, "charge","all")
    
Setting parameters in python

.. code-block:: python

    charge.setParameters(alpha=0.25,r_cut=9.0);


Activating the fix

.. code-block:: python

    #Activate fix
    state.activateFix(charge)



