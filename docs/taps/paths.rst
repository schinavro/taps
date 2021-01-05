.. module:: taps.paths
.. module:: paths

=====
Paths
=====

The :class:`Paths` object is container class that have

Simple line between two object::

    import numpy as np
    from taps.paths import Paths
    coords =              # D x N
    paths = Paths(coords)

In the case of atomic system::

  import numpy as np
  from taps.paths import Paths
  from taps.model import AtomicModel
  coords =              # 3 x A x N
  paths = Paths(coords, model = AtomicModel(symbols=symbols))

Atomic Model uses ASE as a creation for atoms object.
