import os
import numpy as np
from numpy import newaxis as nax
from collections import OrderedDict

from taps.db import PathsDatabase
from taps.pathfinder import PathFinder
from taps.utils.shortcut import isstr, isLst, isdct, isbool
from taps.visualize import view


class GPAOPhase:
    def __init__(self, cov_tol=None, data_ids=None):
        self.cov_tol = cov_tol

    def check_convergence(self, paths, printt):
        cov_max = paths.get_covariance().max()
        printt(" Covmax: %.3f" % cov_max)
        if cov_max < self.cov_tol:
            return 1
        printt(" Number of Dat: %d" % len(paths.imgdata.data_ids))
        return 0

    def acquisition(self, paths, printt):
        idx = np.argmax(paths.get_covariance())
        printt(idx)
        ids = paths.add_data(index=[idx], blackids=paths.imgdata.data_ids)
        return ids


class UncertainOrMaximumEnergy(GPAOPhase):
    def __init__(self, cov_tol=None, Etol=None):
        self.cov_tol = cov_tol
        self.Etol = Etol

    def check_convergence(self, paths, printt):
        cov = paths.get_covariance()
        cov_max = cov.max()
        if cov_max >= self.cov_tol:
            printt('Uncertain covmax: %.3f' % cov_max)
            return 0.
        E = paths.get_potential()
        Emax = E.max()
        Ereal = paths.get_potential(index=[np.argmax(E)], real_model=True)[0]
        if np.abs(Emax - Ereal) >= self.Etol:
            printt('Emax err Emax, Ereal: %.3f, %.3f' % (Emax, Ereal))
            return 0
        printt("Converged! ")
        return 1

    def acquisition(self, paths, printt):
        cov = paths.get_covariance()
        cov_max = cov.max()
        if cov_max < self.cov_tol:
            idx = np.argmax(cov)
            return paths.add_data(index=[idx], blackids=paths.imgdata.data_ids)
        E = paths.get_potential()
        return paths.add_data(index=[np.argmax(E)],
                              blackids=paths.imgdata.data_ids)


class AutoEt(GPAOPhase):
    def __init__(self, Et_tol=None, cov_tol=None, Etol=None, data_ids=None):
        self.Et_tol = Et_tol
        self.cov_tol = cov_tol
        self.Etol = Etol

    def check_convergence(self, paths, printt):
        Et = paths.finder.real_finder.action_kwargs['Energy Restraint']['Et']
        Emax = paths.get_potential().max()
        if Emax < Et:
            printt("Target energy adjusting: Emax %.3f > Et %.3f" % (Emax, Et))
            return 0
        elif (Emax - Et) > 4 * self.Et_tol:
            printt('Target energy not converged %.3f' % (Emax - Et))
            return 0
        return super().check_convergence(paths, printt)

    def acquisition(self, paths, printt, *args, **kwargs):
        Et = self.get_next_et(paths)
        paths.finder.real_finder.action_kwargs['Energy Restraint']['Et'] = Et
        paths.model.mean.hyperparameters = self.get_next_mean(paths)
        return super().acquisition(paths, printt, *args, **kwargs)

    def get_next_et(self, paths, **kwargs):
        Et = paths.finder.real_finder.action_kwargs['Energy Restraint']['Et']
        E = paths.get_potential_energy()

        Emax = np.max(E)
        Emin = np.min(E)
        if Et > Emax:
            Et = Emin
        elif (Emax - Et) <= 2 * self.Et_tol:
            Et = (Emax + Emin) / 2
        else:
            Et = (Emax + Et - 2*self.Et_tol) / 2
        return Et

    def get_next_mean(self, paths, **kwargs):
        E = paths.get_potential_energy()
        return E.max() #  + (V.max() - V.min())*0.1

class GPAO(PathFinder):

    def __init__(self, real_finder=None, restart=False, logfile=None,
                 maxtrial=10, plot_kwargs=None, phase_kwargs=None,
                 pathsdatabase=None, **kwargs):
        super().__init__(**kwargs)
        self.real_finder = real_finder
        self.restart = restart
        self.logfile = logfile
        self.maxtrial = maxtrial
        self.plot_kwargs = plot_kwargs or {}
        self.phase_kwargs = phase_kwargs or {}
        # self.pathsdatabase = pathsdatabase or PathsDatabase()
        self.pathsdatabase = pathsdatabase

    def optimize(self, paths, label=None, real_finder=None, restart=None,
                 logfile=None, maxtrial=None, plot_kwargs=None,
                 phase_kwargs=None, pathsdatabase=None, **kwargs):
        # Initialize
        label = label or self.label or paths.label
        real_finder = real_finder or self.real_finder
        if restart is not None:
            restart = restart
        logfile = logfile or self.logfile
        maxtrial = maxtrial or self.maxtrial
        plot_kwargs = plot_kwargs or self.plot_kwargs
        phase_kwargs = phase_kwargs or self.phase_kwargs
        pathsdatabase = pathsdatabase or self.pathsdatabase
        # LOGGER INIT
        close_log = False
        if logfile is None:
            def printt(*line, end='\n'):
                print(*line, end=end)
        elif isinstance(logfile, str):
            logdir = os.path.dirname(logfile)
            if logdir == '':
                logdir = '.'
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            logfile = open(logfile, 'a')
            def printt(*line, end='\n'):
                lines = ' '.join([str(l) for l in line]) + end
                logfile.write(lines)
                logfile.flush()
            close_log = True
        elif logfile.__class__.__name__ == "TextIOWrapper":
            def printt(*line, end='\n'):
                lines = ' '.join([str(l) for l in line]) + end
                logfile.write(lines)
                logfile.flush()
        # Finish Initialize
        # DEFINE for convinience
        pdb = pathsdatabase
        # End DEFINE

        printt("====================")
        printt("       GPAO")
        printt("====================")
        iteration = 0
        if restart:
            printt("\nReading %s " % pdb.filename)
            paths_data = pdb.get_last_paths_data()
            paths = paths_data['paths']
            iteration = data['iteration'] + 1

        for name, p_kwargs in phase_kwargs.items():
            class_name = name.replace(" ", "")
            module = __import__('taps.pathfinder.gpao', {}, None, [class_name])
            phase = getattr(module, class_name)(**p_kwargs)
            printt("Phase       : %s" % name)
            while True:
                printt("====================")
                printt("       %s : %03d" % (name, iteration))
                printt("====================")
                if maxtrial < iteration:
                    printt("Max iteration, %d, reached! " % maxtrial)
                    break
                if phase.check_convergence(paths, printt):
                    printt("Converged! ")
                    break
                new_ids = phase.acquisition(paths, printt)
                str_ids = [str(id) for id in new_ids]
                printt("Iteration    : %d" % iteration)
                printt("ImgData idx  : %s" % ', '.join(str_ids))
                paths.search(real_finder=True, logfile=printt)
                self.results.update(paths.finder.real_finder.results)
                self._save(paths, iteration=iteration)
                iteration += 1
            printt("")

        if close_log:
            logfile.close()
        return paths

    def _save(self, paths, pathsdatabase=None, plot_kwargs=None, iteration='',
              label=None):
        pathsdatabase = pathsdatabase or self.pathsdatabase
        plot_kwargs = plot_kwargs or self.plot_kwargs
        iteration = iteration
        label = label or self.label

        if plot_kwargs != {}:
            if iteration != '':
                iteration = '_' + '%03d' % iteration
                plot_kwargs['filename'] = label + iteration
            view(paths, **plot_kwargs)
        pathsdatabase.write(data=[{'paths': paths}])

    def _load(self, paths, filename=None, plot_kwargs=None):
        PathsDatabase = PathsDatabase(filename)
        query = "rowid DESC LIMIT 1;"
        where = " ORDER BY "
        columns = ['rowid', 'paths']
        data = PathsDatabase.read(query=query, where=where, columns=columns)[0]
        data['paths'] = self.patch_paths(data['paths'])
        #### IDK WHY but remove read-only
        data['paths'].coords = data['paths'].coords.copy()
        return data
