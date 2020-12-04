import numpy as np
from taps.coords import Coords
from taps.utils.arraywrapper import arraylike


@arraylike
class SineBasis(Coords):
    """
    Sine wave basis representaion of coordinates
    """
    def __init__(self, init=None, fin=None, **kwargs):
        if init is None:
            init = [[0]]
        if fin is None:
            fin = [[0]]
        self.init, self.fin = init, fin

        super().__init__(**kwargs)

    @property
    def Nk(self):
        return self._Nk or self.N

    def __repr__(self):
        return 'SineBasis{}'.format(self.coords.shape)
