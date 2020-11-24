import numpy as np
from taps.visualize.plotter import Plotter


class MullerBrown(Plotter):
    def __init__(self, **kwargs):
        self.calculate_map = True
        self.ttl2d = 'Muller Brown Potential Energy Surface'
        self.xlim2d = np.array([-1.8, 1.2])
        self.ylim2d = np.array([-0.7, 2.3])
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-1.8, 1.2])
        self.ylimGMu = np.array([-0.7, 2.3])
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-1.75, 0.75, 15)
        self.lvlsGMu = np.linspace(-1.75, 0.75, 15)
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.pbc = np.array([False, False])
        self.energy_range = (-1.75, 0.75)
        self.quiver_scale = 3
        self.ylimHE = (-1.5, 0.)
        self.ylimTE = (0, 1)

        super().__init__(**kwargs)
