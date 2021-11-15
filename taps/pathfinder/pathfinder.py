import re
import time
import pickle
import subprocess
import numpy as np
from numpy import newaxis as nax
from numpy.linalg import norm


class PathFinder:

    def __init__(self, results={}, relaxed=None, label=None, prj=None,
                 **kwargs):
        self.results = results
        self.relaxed = relaxed
        self.label = label
        self.prj = prj
        self._cache = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        if key == 'real_finder':
            return self
        else:
            super().__getattribute__(key)

    @property
    def label(self):
        if self.directory == '.':
            return self.prefix

        if self.prefix is None:
            return self.directory + '/'

        return '{}/{}'.format(self.directory, self.prefix)

    @label.setter
    def label(self, label):
        if label is None:
            self.directory = '.'
            self.prefix = None
            return

        tokens = label.rsplit('/', 1)
        if len(tokens) == 2:
            directory, prefix = tokens
        else:
            assert len(tokens) == 1
            directory = '.'
            prefix = tokens[0]
        if prefix == '':
            prefix = None
        self.directory = directory
        self.prefix = prefix

    def search(self, paths=None, real_finder=False, pbs=None, **kwargs):
        """
        pbs : dictionary
        """
        if real_finder:
            finder = self.real_finder
        else:
            finder = self

        finder.optimize(paths=paths, **kwargs)

    def isConverged(self, *args, **kwargs):
        return True
