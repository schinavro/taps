


class SineBasis(Coordinates):

    @property
    def dt(self):
        self.epoch / self.N
