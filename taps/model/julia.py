import os
import numpy as np
import subprocess

from taps.model import Model


class Julia(Model):
    """ Externel calculator for parallel calculation
    """
    model_parameters = {}

    def __init__(self,
                 model_model=None,
                 model_label=None,
                 model_potential_unit=None,
                 model_data_ids=None,
                 model_prj=None,
                 model_kwargs=None,
                 coords_epoch=None,
                 coords_unit=None,
                 # finder_finder=None,
                 # finder_prj=None,
                 # finder_label=None,
                 **kwargs):
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)

        self.model_model = model_model
        self.model_label = model_label
        self.model_potential_unit = model_potential_unit
        self.model_data_ids = model_data_ids
        self.model_prj = model_prj
        self.model_kwargs = model_kwargs
        self.coords_epoch = coords_epoch
        self.coords_unit = coords_unit
        # self.finder_finder = finder_finder
        # self.finder_prj = finder_prj
        # self.finder_label = finder_label

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=['potential'],
                  model_model=None,
                  model_label=None,
                  model_potential_unit=None,
                  model_data_ids=None,
                  model_prj=None,
                  model_kwargs=None,
                  coords_epoch=None,
                  coords_unit=None,
                  finder_finder=None,
                  finder_prj=None,
                  finder_label=None,
                  **kwargs):
        model_model = model_model or self.model_model
        assert model_model is not None, 'Set model_model'
        model_label = model_label or self.model_label or self.label or \
            paths.label or 'julia_model'
        model_potential_unit = model_potential_unit or \
            self.model_potential_unit or paths.model.potential_unit
        model_data_ids = model_data_ids or self.model_data_ids or self.data_ids
        model_prj = model_potential_unit or self.model_prj or \
            self.prj.__class__.__name__
        model_kwargs = model_kwargs or self.model_kwargs or kwargs
        coords_epoch = coords_epoch or self.coords_epoch or paths.coords.epoch
        coords_unit = coords_unit or self.coords_unit or paths.coords.unit
        finder_finder = finder_finder or self.finder_finder or \
            paths.finder.__class__.__name__
        finder_prj = finder_prj or self.finder_prj or paths.real_finder.prj
        finder_label = finder_label or self.finder_label or \
            paths.real_finder.label or paths.label or 'julia_finder'

        modeljl = '/home/schinavro/libCalc/taps/julia/model.jl'
        if model_label[0] == '/':
            filename = model_label + '.npz'
        else:
            filename = os.getcwd() + '/' + model_label + '.npz'
        cds = paths.coords
        np.savez(filename,
                 coords=coords,
                 properties=properties,
                 model_model=model_model,
                 model_label=model_label,
                 model_potential_unit=model_potential_unit,
                 model_prj=model_prj,
                 model_kwargs=model_kwargs,
                 model_real_kwargs=model_real_kwargs,
                 coords_epoch=coords_epoch,
                 coords_unit=coords_unit,
                 finder_finder=finder_finder,
                 finder_prj=finder_prj,
                 finder_label=finder_label,
                 **kwargs)

        command = 'julia --project %s %s' % (modeljl, filename)
        # command = 'mpiexec ' + command
        con = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=64)
        while line := con.stdout.readline():
            print(line, end='')
        results = np.load('result.npz')

        self.results = results
