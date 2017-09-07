External Harmonic Potential Fix
===============================

Overview
^^^^^^^^

Applies an external potential to a group of atoms to mimic confinement in a harmonic well. 

.. math::
    U_{\text{ext}} = \frac{1}{2} \sum_{d=x,y,z} k_d(r_d - r_{d0})^2,

where :math:`\mathbf{k}=(k_x,k_y,k_z), \mathbf{r}_0=(r_{x0},r_{y0},r_{z0})` are ``Vector`` parameters that must be set to define the harmonic well. As the parameters are vectors, one- and two-dimensional harmonic potentials may be created by setting some of the spring coefficients to zero.

The harmonic potentials are defined directly within the python input script as a ``Fix`` using its constructor.

Constructor
^^^^^^^^^^^
.. code-block:: python

    FixExternalHarmonic(state,handle,groupHandle,k,r0)

Arguments

``state``
    Simulation state to apply the fix.

``handle``
    A name for the object.  Named argument.

``groupHandle``
    Group of atoms to which the fix is applied.  Named argument.  Defaults to ``all``.

``k``
    Vector of spring constant coefficient for :math:`k_x,k_y,k_z`  for strength of confining potential in each dimension. These are specified as ``Vector(kx,ky,kz)``.

``r0``
    Vector of minima :math:`r_{x0},r_{y0},r_{z0}` in each dimension for the harmonic potential. These are specified as ``Vector(rx,ry,rz)``.

Examples
^^^^^^^^
Adding and activating the fix

.. code-block:: python

    #Add Fix for external harmonic potential
    UextHarm = FixExternalHarmonic(state,handle='Uharm',groupHandle='all',
      k = Vector(1.0,0.0,0.5), r0 = Vector(0.0,0.0,1.0))
    #Activate fix
    state.activateFix(UextHarmt)
