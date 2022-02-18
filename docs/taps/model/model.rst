.. module:: taps.models.models

=====
Model
=====

:class:`Model` calculate static property of given :attr:`coords`.
Static properties including potentials, gradients and hessian, is calculated using :class:`Model` class.

Example
=======

   import numpy as np
   from taps.paths import Paths
   from taps.models import MullerBrown
   paths = Paths()
   coords = [np.linspace(-0.558, 0.623, 15), np.linspace(1.44, 0.028, 15)]
   paths.coords = coords
   paths.model = MullerBrown()
   paths.get_potential()

you can use :attr:`index` keyword to calculate a part of potential.
By default, :attr:`index` is set to :attr:`np.s_[1:-1]`, meaning it will calculate intermediate images without initial and final state.
::

>>> paths.get_potential(index=[0, -1])  # Potential of initial and final state

Model as a super class
======================

Every calculator model should inherits the :class:`Model` class.
This is due to additional manipulation before it goes to real calculator.
In the super class, Model conduct 3 additional work.

Cache
-----

First it saves the calculation history if :attr:`cache` is enabled.
While saving it checks current calculation of given :attr:`coords` is exists or not.

Suppose a user calculate the forces of the system and potential. ::

>>> forces = paths.get_forces()        # We know the potential of system
>>> potential = paths.get_potential()  # However, recalculate

For a simple model calculation, it would not be a huge problem.
But a calculation with hybrid functional, that is too much waste.
In such case, user can manually ask calculator to save the all results while calculating forces.::

>>> forces = paths.get_forces(caching=True)  # Save all results from calculation, including potential data
>>> potential = paths.get_potential(caching=True)   # Read the potential from cache



Model as a subclass of Model
============================

Every model class should contain `calculate` that gets coordinates and save it on the dictionary


.. toctree::

    mullerbrown
    ase


List of all Methods
===================

.. autoclass:: Model
  :members:

