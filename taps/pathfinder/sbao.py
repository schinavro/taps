import os
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
        if label[0] == '/':
            filename = label + '.npz'
        else:
            filename = os.getcwd() + '/' + label + '.npz'
        # command = 'mpiexec julia --project %s %s' % (sbaojl, filename)
        command = 'julia --project %s %s' % (sbaojl, filename)
        cds = paths.coords
        epoch, Nk, D, init, fin = cds.epoch, cds.Nk, cds.D, cds.init, cds.fin
        np.savez(filename, epoch=epoch, Nk=Nk, D=D, init=init.T,
                 fin=fin.T, ak=cds.rcoords.T)
        con = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
        while line := con.stdout.readline():
            print(line, end='')
        # self.print_file(file=outfile, running=isRunning, process=p)
        out = np.load('result.npz')
        ak = out['resk']
        paths.coords.rcoords = ak.T
        return paths.copy()
