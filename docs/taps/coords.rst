.. module:: taps.coords.cartesian
.. _ase: https://wiki.fysik.dtu.dk/ase/index.html

=========
Cartesian
=========
Coordinate representaion of a pathway.

Array Order
===========

.. figure:: images/array_order.png

Since numpy array uses C like array indexing, we indexed the intermediate image at the last rank.
For example, suppose we are creating a pathway on the 2D Múller Brown potential having 15 steps between initial and final state.

    >>> import numpy as np
    >>> from taps.paths import Paths
    >>> paths = Paths()
    >>> coords = [np.linspace(-0.558, 0.623, 15), np.linspace(1.44, 0.028, 15)]
    >>> paths.coords = coords
    >>> print(paths.coords.shape)
    (2, 15)

Calculate Kinetic energy
========================

:class:`Cartesian` contains tools for calculating kinetic property of the pathway. For example,

    >>> paths.get_kinetic_energies()

Since the way of calculating kinetic energy entirly depends on the coordinate representation, way of getting kinetic energy is differ with individual coords.
Here, we set cartesian coordinate as a default representation.

Cartesian representation
========================

For the atomic system in the cartesian coordinates, we use 3 rank representation where each image is indexed at the last rank.
In the atomic representation, `ASE <https://wiki.fysik.dtu.dk/ase/index.html>`_ can be good tools for building such system.
For example, suppose a system having 14 atoms with 300 images between initial and final state, coordinate representation is

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

atomic representation is array like object with shape :math:`3 \times A \times N` where :math:`A` is number of atoms and :math:`N` is the number of intermediate images.

Array like
==========

Cartesian is an arraylike object. It would be easier to consider it a numpy array with additional kinetic calculation method. To return only the array,

   >>> coords_array = paths.coords[..., :]

If you want to keep the class, but send partial info you can call the coords.

   >>> coords_copied = paths.coords(index=np.s_[:])


List of all Methods
===================

.. autoclass:: Cartesian
   :members:
   :exclude-members: Nk,T, all, any, argmax,argmin,argpartition,argsort,astype,base,byteswap,choose,clip,compress,conj,conjugate,ctypes,cumprod,cumsum,data,diagonal,dot,dtype,dump,dumps,fill,flags,flatte,getfield,imag,item,itemset,itemsize,max,mean,min,nbytes,ndim,newbyteorder, nonzero, partition, prod ,ptp ,put ,ravel ,rcoords, repeat , reshape , resize , round , searchsorted , setfield , setflags , shape , size , sort , squeeze , std , strides , sum , swapaxes , take , tobytes , tofile , tolist , tostring , trace , transpose , var , view
