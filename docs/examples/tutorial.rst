========
Tutorial
========

Muller Brown potential
======================

>>> from taps.paths import Paths
>>> from taps.model.mullerbrown import MullerBrown
>>> paths = Paths()
>>> paths.model = MullerBrown()

>>> import numpy as np
>>> N = 300
>>> x = np.linspace(-0.55822365, 0.6234994, N)
>>> y = np.linspace(1.44172582, 0.02803776, N)
>>> paths.coords = np.array([x, y])

.. figure:: images/tutorial_mb.png

>>> from taps.pathfinder import DAO
>>> from taps.projector import Sine
>>> paths.finder = DAO(action_name=['Onsager Machlup', 'Energy Conservation'], muE=1, Et_type='manual', sin_search=False, Et=-0.41, prj = Sine(init=paths.coords[..., 0], fin=paths.coords[..., -1], N=N, Nk=N-2))
>>> paths.coords.epoch = 6
>>> paths.search()
Action name  :  Onsager Machlup + Energy Conservation
Target energy:  -0.41
Target type  :  manual
muE          :  1
gamma        :  1
            Iter   nfev   njev        S   dS_max
Max out :    500    520    520   2.0785   2.6790
Max out :   1000   1024   1024   1.8292   0.5648
Max out :   1500   1530   1530   1.7684   0.6026
Max out :   2000   2035   2035   1.6990   0.4721
Max out :   2500   2541   2541   1.6406   0.3597
Success :   2509   2556   2556   1.6401   0.0937
Success :   2509   2557   2557   1.6401   0.0937


>>> from taps.visualize import view
>>> viewer = view(paths, calculate_map=True, viewer='MullerBrown')
.. figure:: images/tutorial_mb2.png
