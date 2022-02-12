import re
import time
import pickle
import subprocess
import numpy as np
from numpy import newaxis as nax
from numpy.linalg import norm

from taps.projectors import Projector


class PathFinder:

    def __init__(self, results=None, relaxed=None, label=None, prj=None,
                 _cache=None):
        self.results = results or dict()
        self.relaxed = relaxed
        self.label = label
        self.prj = prj or Projector()
        self._cache = _cache or dict()

    def __getattr__(self, key):
        if key == 'real_finder':
            return self
        else:
            super().__getattribute__(key)

    def search(self, paths=None, real_finder=False, **kwargs):
        if real_finder:
            finder = self.real_finder
        else:
            finder = self

        finder.optimize(paths=paths, **kwargs)

    def isConverged(self, *args, **kwargs):
        return True

    def get_x0(self, coords):
        return coords.flatten()
