.. module:: taps.paths
.. _settingup:

=====
Paths
=====
The :class:`Paths` object is container class that handles :meth:`io` or bridging to other module.

Overall Structure
=================

.. figure:: <where is image>
   :alt: Jubler con
   :scale: 40 %
   Getting Jubler to use MPlaer

:class:`Paths` class contains modularized classes :class:`Coords`, :class:`Projector`, :class:`Model`, :class:`Pathfinder` and etc.
:class:`Paths` interfaces between modularized classes by sending :attr:`self`, the pointer of self (:attr:`paths`), when :attr:`paths` call the function of modules.
Each subcontained class recieves :class:`Paths` as an instance and uses that to access the other modules.

For example, :class:`Model` needs :class:`Coords` information when it calculate potential.
When user call :meth:`get_potential_energy()` in a :attr:`paths`,

>>> paths.get_potential_energy()

:class:`Paths` simply calls the :meth:`get_potential_energy()` in the :class:`Model`. When calling it, :class:`Paths` sends additional pointer :attr:`self` to it::

  # paths.py
  ...
  def get_potential_energy(self, ...):
      return self.model.get_potential_energy(self, ...)
  ...

:class:`Model` recieves :class:`Paths` as a :attr:`paths` instance and use it to access :attr:`coords`::

  # model.py
  def get_properties(self, paths, ...):
      ...
      coords = paths.coords
      ...

That is, simply put, :class:`Paths` act as a interface between modules.
This way, modules can access each module without additional creation of instance.
However, user may wants to call :attr:`paths` only when it necessary since it is prone to trap in infinite loop.

Load & Save
===========

There can be a lots of approach to save the python class.
Since :class:`Paths` contains multiple class in the class, we took simple approach.
:class:`pickle` saves the class instance magically by recursively serialize every attribute of the contained instance.
This makes saving simply

   >>> import pickle
   >>> with open('filename', 'w') as f:
   >>>     pickle.dump(paths, f)

and make load simply

   >>> with open('file', 'r') as f:
   >>>     paths = pickle.load(f)

However, It has some security issue when it read the serialized file.
It would be better to separate every instance that needs to be saved and only serialize that with more secure method, like using :class:`json`.
Currently, we implemented dictionary contains info needs to be saved and check it is valid info or not, but detailed implementation is not yet implemented.


List of all Methods
===================
.. autoclass:: Paths
    :members:
    :no-undoc-members:
