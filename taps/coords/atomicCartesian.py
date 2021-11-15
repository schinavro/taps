from taps.coords import Cartesian
from taps.utils.arraywrapper import arraylike

@arraylike
class AtomicCartesian(Cartesian):

    def __int__(self, mass=None, **kwargs):
        self.mass = mass
        super().__init__(**kwargs)


    def similar(self):
        return self.__class__(coords=None, unit=self.unit, epoch=self.epoch,
                              mass=self.mass)

    def mass(self):
        return self.mass
