import numpy as np
import subprocess

from taps.pathfinder import PathFinder


class SBAO(PathFinder):
    finder_parameters = {}

    def __init__(self, **kwargs):
        super().finder_parameters.update(self.finder_parameters)
        self.finder_parameters.update(super().finder_parameters)

        super().__init__(**kwargs)

    def optimize(self, paths=None, logfile=None, Et=None, Et_type=None,
                 action_name=None, label=None, **args):
        label = label or self.label or paths.label or 'sbao'
        sbaojl = '/home/schinavro/libCalc/taps/taps/pathfinder/sbao/sbao.jl'
        command = 'julia --project %s %s' % (sbaojl, label + '.npz')
        cds = paths.coords
        epoch, Nk, D, init, fin = cds.epoch, cds.Nk, cds.D, cds.init, cds.fin
        np.savez(label + '.npz', epoch=epoch, Nk=Nk, D=D, init=init.T,
                 fin=fin.T, ak=cds[..., :].T)
        subprocess.Popen(command, shell=True, universal_newlines=True)
        # self.print_file(file=outfile, running=isRunning, process=p)
        out = np.load('result.npz')
        ak = out['ak']
        paths.coords = ak
