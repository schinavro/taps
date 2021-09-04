import numpy as np
from taps.visualize.plotter import Plotter

class Alaninedipeptide(Plotter):
    def __init__(self, **kwargs):
        self.mapfile = "alanine_map.pkl"
        self.calculate_map = False
        self.ttl2d = 'Alaninedipeptide PES'
        self.xlim2d = np.array([-180., 180.]) * np.pi / 180
        self.ylim2d = np.array([-180., 180.]) * np.pi / 180
        self.cmp2d = 'cividis'
        self.alp2d = 0.5
        self.xlimGMu = np.array([-180., 180.]) * np.pi / 180
        self.ylimGMu = np.array([-180., 180.]) * np.pi / 180
        self.cmpGMu = 'cividis'
        self.lvls2d = np.linspace(-130.0, -129.0, 15)
        self.lvlsGMu = np.linspace(-130.0, -129.0, 15)
        self.cmpGCov = 'plasma'
        self.lvlsGCov = np.linspace(0, 2, 15)
        self.pbc = np.array([False, False])
        self.energy_range = (-130.0, -129.0)
        self.quiver_scale = 3
        self.ylimHE = (-130.0, -129.0)
        self.ylimTE = (0, 1)
        self.conformation = 180 / np.pi

        super().__init__(**kwargs)
