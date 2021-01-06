.. module:: taps.coords
.. _ase: https://wiki.fysik.dtu.dk/ase/index.html

======
Coords
======
Coordinate representaion of a pathway.

Array Order
===========

.. figure:: images/array_order.png

Since numpy array uses C like array indexing, we index the image at the last rank.
For example, suppose we are creating a pathway on the 2D Múller Brown potential having 15 steps between initial and final state.::

    >>> import numpy as np
    >>> from taps.paths import Paths
    >>> paths = Paths()
    >>> coords = [np.linspace(-0.558, 0.623, 15), np.linspace(1.44, 0.028, 15)]
    >>> paths.coords = coords
    >>> print(paths.coords.shape)
    (2, 15)

Cartesian representation
========================

For the atomic system in the cartesian coordinates, we use 3 rank representation where each image is indexed at the last rank.
In the atomic representation, `ASE <https://wiki.fysik.dtu.dk/ase/index.html>`_ can be good tools for building such system.
For example, suppose a system having 22 atoms with 300 images between initial and final state, coordinate representation is ::

    >>> import numpy as np
    >>> from ase.build import fcc100, add_adsorbate
    >>> slab = fcc100('Al', size=(2, 2, 3))
    >>> add_adsorbate(slab, 'Au', 1.7, 'hollow')
    >>> slab.center(axis=2, vacuum=4.0)
    >>> init = slab.positions.T.copy()  # shape 3x14
    >>> slab[-1].x += slab.get_cell()[0, 0] / 2
    >>> fin = slab.positions.T.copy()   # shape 3x14
    >>> dist = (fin -  init) # 3 x 14
    >>> N = 300
    >>> coords = init[:, np.newaxis] + np.linspace(0, 1, N)[np.newaxis, :] * dist[:, np.newaxis]
    >>> print(coords.shape)
    (3, 14, 300)

atomic representation is array like object with shape 3 x A x N where A is
number of atoms and N is the number of intermediate images including inital
and final.

Array like
==========

Coords is arraylike object.

List of all Methods
===================

.. autoclass:: Coords
   :members:
