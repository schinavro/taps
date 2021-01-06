.. module:: taps.paths
.. _settingup:

=====
Paths
=====
The :class:`Paths` object is container class that handles :meth:`io` or bridging to other module.
:ref:`akfdhaef <settingup>`

Overall Structure
=================

.. figure:: <where is image>
   :alt: Jubler con
   :scale: 40 %
   Getting Jubler to use MPlaer

:class:`Paths` class contains modularized classes :class:`Coords`, :class:`Projector`, :class:`Model`, :class:`Pathfinder` and etc.
:class:`Paths` interfaces between these classes by sending :attr:`self` pointer to each class.
Each subcontained class recieves :class:`Paths` as instance and uses that to access to the other class
For example, :class:`Model` needs :class:`Coords` information when it calculate potential.
When user call :attr:`paths` a :meth:`get_potential_energy()`,

>>> paths.get_potential_energy()

:class:`Paths` simply calls the :meth:`get_potential_energy()` in the :class:`Model` attatching :attr:`self` to it::

  ...
  def get_potential_energy(self, ...):
      return self.model.get_potential_energy(self, ...)
  ...

and in the :class:`Model` class, use that :attr:`paths` instance to access the :attr:`coord`::

  def get_properties(self, paths, ...):
      ...
      coords = paths.coords
      ...

That is, simply put, :class:`Paths` act as a interface between modules.

Load & Save
===========

Saving and loading the current pathway instance is necessary.
Since :class:`Paths` contains other multiple module, we used more simple approach.
:module:`pickle` helps serialize python instance contains other class.
It has some security issue when it read the serialized file.
It would be better to separate every instance that needs to be saved and only serialize that with more secure method, like using :module:`json`.
Currently, we implemented dictionary contains info needs to be saved and check it is valid info or not, but detail implementation can be done later.


List of all Methods
===================
.. autoclass:: Paths
    :members:
