External Quartic Potential Fix
==============================

Overview
^^^^^^^^

Applies an external potential to a group of atoms to mimic confinement in a quartic/double well potential. 

.. math::
    U_{\text{ext}} = \sum_{i=1}^4 \mathbf{k}_i\cdot(\Delta \mathbf{r})^i, 

where :math:`\mathbf{k}_i=(k_{ix},k_{iy},k_{iz}) \text{for} i = 1,\dots,4, \Delta \mathbf{r} = (\mathbf{r} - \mathbf{r}_0)` for :math:`\mathbf{r}_0=(r_{x0},r_{y0},r_{z0})`. Here, each of the :math:`\mathbf{k}_i` and :math:`\mathbf{r}_0`  are ``Vector`` parameters that must be set to define the external potential. As the parameters are vectors, one- and two-dimensional potentials may be created by setting some of the spring coefficients to zero.

Quartic potentials are defined directly within the python input script as a ``Fix`` using its constructor.

Constructor
^^^^^^^^^^^
.. code-block:: python

    FixExternalQuartic(state,handle,groupHandle,k1,k2,k3,k4,r0)

Arguments

``state``
    Simulation state to apply the fix.

``handle``
    A name for the object.  Named argument.

``groupHandle``
    Group of atoms to which the fix is applied.  Named argument.  Defaults to ``all``.

``k1``
    Vector of spring constant coefficient for :math:`k_{1x},k_{1y},k_{1z}`  for strength of linear portion of confining potential in each dimension. These are specified as ``Vector(k1x,k1y,k1z)``.

``k2``
    Vector of spring constant coefficient for :math:`k_{2x},k_{2y},k_{2z}`  for strength of quadratic portion of confining potential in each dimension. These are specified as ``Vector(k2x,k2y,k2z)``.

``k3``
    Vector of spring constant coefficient for :math:`k_{3x},k_{3y},k_{3z}`  for strength of cubic portion of confining potential in each dimension. These are specified as ``Vector(k3x,k3y,k3z)``.

``k4``
    Vector of spring constant coefficient for :math:`k_{4x},k_{4y},k_{4z}`  for strength of quartic portion of confining potential in each dimension. These are specified as ``Vector(k4x,k4y,k4z)``.

``r0``
    Vector of minima :math:`r_{x0},r_{y0},k_{z0}` in each dimension for the harmonic potential. These are specified as ``Vector(rx,ry,rz)``.

Examples
^^^^^^^^
Adding and activating the fix

.. code-block:: python

    #Add Fix for external quartic potential
    UextDblWell = FixExternalQuartic(state,handle='Udbl',groupHandle='all',
      k1 = Vector(0.0,0.0,0.0),
      k2 = Vector(-1.,-1.,-1.),
      k3 = Vector(0.0,0.0,0.0),
      k4 = Vector(0.015,0.015,0.015),
      r0 = Vector(0.0,0.0,1.0))
    #Activate fix
    state.activateFix(UextDblWell)
