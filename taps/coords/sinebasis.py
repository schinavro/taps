import numpy as np
from scipy.fftpack import dst, idst
from taps.coords import Coords
from taps.utils.arraywrapper import arraylike


@arraylike
class SineBasis(Coords):
    """
    Sine wave basis representaion of coordinates
    """

    def __repr__(self):
        return 'SineBasis{}'.format(self.coords.shape)

    @property
    def rcoords(self):
        """ Recieprocal representation of a pathway
            Returns rcoords array
            D x (N - 2) array
        """
        curve = self.coords - self.line
        return dst(curve[..., 1:-1], type=1)

    @rcoords.setter
    def rcoords(self, rcoords):
        """
            For numerical purpose, set rpath.
            D x M x (P - 2)  array
        """
        line, norm = self.line, self.sine_norm
        self.coords[..., 1:-1] = line[..., 1:-1] + idst(rcoords, type=1) / norm

    @property
    def init(self):
        return self.coords[..., 0, np.newaxis]

    @property
    def fin(self):
        return self.coords[..., -1, np.newaxis]

    @property
    def line(self):
        if self.__dict__.get('_line') is None:
            init, fin, N = self.init, self.fin, self.N
            dir = (fin - init)
            line = init + dir * np.linspace(0, 1, N)
            self._line = line
        return self._line

    @property
    def sine_norm(self):
        if self.__dict__.get('_sine_norm') is None:
            self._sine_norm = 2 * (self.N - 1)
        return self._sine_norm
