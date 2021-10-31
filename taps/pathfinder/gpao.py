import os
import numpy as np
from numpy import newaxis as nax
from collections import OrderedDict

from taps.db.data import PathsData
from taps.pathfinder import PathFinder
from taps.utils.shortcut import isstr, isLst, isdct, isbool
from taps.visualize import view


class GPAO(PathFinder):

    finder_parameters = {
        'real_finder': {'default': '"DAO"', 'assert': 'True'},
        'log': {'default': 'None', 'assert': isstr},
        'phases': {'default': '["Maximum uncertainty", "Alternate energy"]',
                   'assert': isLst},
        'convergence_checker': {'default': 'None', 'assert': isLst},
        'phase': {'default': 'None', 'assert': 'True'},
        'gptol': {'default': '0.1', 'assert': 'True'},
        'maxtrial': {'default': '50', 'assert': 'True'},
        'cov_max_tol': {'default': '0.05', 'assert': 'True'},
        'E_max_tol': {'default': '0.05', 'assert': 'True'},
        'distance_tol': {'default': '0.05', 'assert': 'True'},
        'plot': {'default': "view()", 'assert': 'True'},
        'plot_kwargs': {'default': 'dict()', 'assert': isdct},
        'restart': {'default': 'False', 'assert': isbool},
        'last_checker': {'default': "'auto et2'", 'assert': 'True'}
    }

    display_map_parameters = OrderedDict({})
    display_graph_parameters = OrderedDict({})
    display_graph_title_parameters = OrderedDict({
        '_cov_max': {
            'label': r'$\Sigma^{{(max)}}_{{95\%}}$', 'isLaTex': True,
            'under_the_condition':
                # "True",
                "{pf:s}.__dict__.get('_maximum_uncertainty_checked', False)",
            'unit': "{p:s}.model.potential_unit",
            'value': "{pf:s}.{key:s}",
            'kwargs': "{'fontsize': 13}"
        },
        '_muErr': {
            'label': r'$\mu_{{err}}^{{(max)}}$', 'isLaTex': True,
            'under_the_condition':
                "{pf:s}.__dict__.get('_maximum_energy_checked', False)",
            'unit': "{p:s}.model.potential_unit",
            'value': "{pf:s}.{key:s}",
            'kwargs': "{'fontsize': 13}"
        },
        '_mu_Et': {
            'label': r'$\mu^{{(max)}}-E_{{t}}$', 'isLaTex': True,
            'under_the_condition':
                "{pf:s}.__dict__.get('_target_energy_checked', False)",
            'unit': "{p:s}.model.potential_unit",
            'value': "{pf:s}.{key:s}",
            'kwargs': "{'fontsize': 13}"
        },
        'deltaMu': {
            'label': r'$\left|\Delta\mu^{{(max)}}\right|$', 'isLaTex': True,
            'under_the_condition':
                "'maximum mu' == {pf:s}.Phase.lower() and "
                "len({pf:s}.__dict__.get('_Emaxlst', [])) > 1",
            'unit': "{p:s}.model.potential_unit",
            'value': "np.abs({pf:s}._Emaxlst[-1] - {pf:s}._Emaxlst[-2])",
            'kwargs': "{'fontsize': 13}"
        },
    })

    def __init__(self, real_finder=None, log=None, gptol=None,
                 cov_max_tol=None, E_max_tol=None, maxtrial=None, phase=0,
                 phases=None, last_checker=None, distance_tol=None,
                 plot=view, plot_kwargs=None, restart=False,
                 _pbs_walltime="walltime=48:00:00", **kwargs):
        self.real_finder = real_finder
        self.finder_parameters.update(self.real_finder.finder_parameters)
        self.display_map_parameters.update(
            self.real_finder.display_map_parameters)
        self.display_graph_parameters.update(
            self.real_finder.display_graph_parameters)
        self.display_graph_title_parameters.update(
            self.real_finder.display_graph_title_parameters)
        self.log = log
        self.gptol = gptol
        self.cov_max_tol = cov_max_tol
        self.E_max_tol = E_max_tol
        self.distance_tol = distance_tol
        self.plot = plot
        self.plot_kwargs = plot_kwargs
        self.restart = restart
        self.maxtrial = maxtrial
        self.phase = phase
        self.phases = phases
        self.last_checker = last_checker
        self._E = []
        self._cov = []
        self._pbs = False
        self._write_pbs = True
        self._write_pbs_only = False
        self._pbs_walltime = _pbs_walltime
        super().__init__(**kwargs)

    def __getattr__(self, key):
        if key in self.__dict__.get('real_finder', {}).finder_parameters.keys():
            return getattr(self.__dict__['real_finder'], key)
        elif key == 'convergence_checker':
            return self.__dict__['phases']
        else:
            super().__getattribute__(key)

    @property
    def Phase(self):
        if self.phase >= len(self.convergence_checker):
            return self.last_checker
        return self.convergence_checker[self.phase]

    @property
    def Phases(self):
        return self.convergence_checker

    def maximum_uncertainty(self, paths, gptol=None, iter=None):
        cov = paths.get_covariance()
        self._maximum_uncertainty_checked = True
        # self._cov_max = cov.max() / paths.model.hyperparameters['sigma_f']
        self._cov_max = cov.max()
        return np.argmax(cov)

    def check_maximum_uncertainty_convergence(self, paths, idx=None, **kwargs):
        blackids = paths.model.data_ids['image']
        if not self._maximum_uncertainty_checked:
            paths.add_data(index=idx, blackids=blackids)
            return 0
        paths.add_data(index=idx, blackids=blackids)
        if self._cov_max < self.cov_max_tol:
            return 1
        return 0

    def maximum_energy(self, paths, gptol=None, iter=None):
        E = paths.get_potential_energy(index=np.s_[:])
        self._maximum_energy_checked = True
        self._E_max = E.max()
        return np.argmax(E)

    def check_maximum_energy_convergence(self, paths, idx=None, **kwargs):
        blackids = paths.model.data_ids['image']
        if not self._maximum_energy_checked:
            paths.add_data(index=idx, blackids=blackids)
            return 0
        data_ids = paths.add_data(index=idx, blackids=blackids)
        imgdata = paths.get_data(data_ids=data_ids)
        self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        if self._muErr < self.E_max_tol:
            return 1
        return 0

    def uncertain_or_maximum_energy(self, paths, iter=None, **kwargs):
        cov = paths.get_covariance(index=np.s_[:])
        self._maximum_uncertainty_checked = True
        # self._cov_max = cov.max() / paths.model.hyperparameters['sigma_f']
        self._cov_max = cov.max()
        if self._cov_max > self.cov_max_tol:
            return np.argmax(cov)
        E = paths.get_potential_energy(index=np.s_[:])
        self._maximum_energy_checked = True
        self._E_max = E.max()
        return np.argmax(E)

    def check_uncertain_or_maximum_energy_convergence(self, paths, idx=None,
                                                      **kwargs):
        blackids = paths.model.data_ids['image']
        if self._cov_max > self.cov_max_tol:
            data_ids = paths.add_data(index=idx, blackids=blackids)
            return 0
        data_ids = paths.add_data(index=idx, blackids=blackids)
        imgdata = paths.get_data(data_ids=data_ids)
        self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        if self._muErr < self.E_max_tol:
            return 1
        return 0

    def manual_et(self, paths, **kwargs):
        return self.uncertain_or_maximum_energy(paths, **kwargs)

    def check_manual_et_convergence(self, paths, idx=None, **kwargs):
        return self.check_maximum_energy_convergence(paths, idx=idx, **kwargs)

    def auto_et(self, paths, **kwargs):
        paths.finder.real_finder.Et = self.get_next_et(paths)
        paths.finder.real_finder.Et_type = 'manual'
        return self.uncertain_or_maximum_energy(paths, **kwargs)

    def check_auto_et_convergence(self, paths, idx=None, **kwargs):
        blackids = paths.model.data_ids['image']

        V = paths.get_potential_energy(index=np.s_[1:-1])
        data_ids = paths.add_data(index=idx, blackids=blackids)
        imgdata = paths.get_data(data_ids=data_ids)
        if self._maximum_energy_checked:
            self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        # @@@@@@@@@@@@@@@@@@@@
        cov = paths.get_covariance(index=np.s_[:])
        # self._cov_max = cov.max() / paths.model.hyperparameters['sigma_f']
        self._cov_max = cov.max()
        if np.abs(V.max() - paths.finder.real_finder.Et) < self.Et_opt_tol and \
                self._cov_max < self.cov_max_tol:
            return 1
        # @@@@@@@@@@@@@@@
        # if np.abs(V.max() - paths.finder.real_finder.Et) < self.Et_opt_tol and \
        #         self._cov_max > self.cov_max_tol:
        #     return 1
        return 0

    def auto_et2(self, paths, **kwargs):
        paths.finder.real_finder.Et = self.get_next_et(paths)
        paths.finder.real_finder.Et_type = 'manual'
        paths.model.mean.Em = self.get_next_mean(paths)
        paths.model.mean.type = 'manual'

        idx = self.uncertain_or_maximum_energy(paths, **kwargs)
        return idx

    def check_auto_et2_convergence(self, paths, idx=None, **kwargs):
        V = paths.get_potential_energy(index=np.s_[1:-1])
        blackids = paths.model.data_ids['image']
        data_ids = paths.add_data(index=idx, blackids=blackids)
        imgdata = paths.get_data(data_ids=data_ids)
        if self._maximum_energy_checked:
            self._muErr = np.abs(self._E_max - imgdata['V'][-1])
        cov = paths.get_covariance(index=np.s_[:])
        self._cov_max = cov.max()
        cur_Et = paths.finder.real_finder.Et
        Vmax = V.max()
        if Vmax < cur_Et:
            return 0
        elif self._cov_max > self.cov_max_tol:
            return 0
        elif (Vmax - cur_Et) < 4 * self.Et_opt_tol:
            return 1
        return 0

    def maximum_mu(self, paths, **kwargs):
        return self.uncertain_or_maximum_energy(paths, **kwargs)

    def check_maximum_mu_convergence(self, paths, idx=None, **kwargs):
        blackids = paths.model.data_ids['image']
        E = paths.get_potential_energy(index=np.s_[:])
        self._Emaxlst = self.__dict__.get('_Emaxlst', [])
        self._Emaxlst.append(np.max(E))
        if len(self._Emaxlst) < 3:
            paths.add_data(index=idx, blackids=blackids)
            return 0
        paths.add_data(index=idx, blackids=blackids)
        V0, V1 = self._Emaxlst[-2:]
        if np.abs(V0 - V1) < self.Et_opt_tol:
            return 1
        return 0

    def get_next_et(self, paths, **kwargs):
        V = paths.get_potential_energy(index=np.s_[1:-1])
        self._target_energy_checked = True
        self._mu_Et = V.max() - paths.finder.real_finder.Et

        maxV = np.max(V)
        minV = np.min(V)
        if paths.finder.real_finder.Et > maxV:
            Et = minV
        elif (maxV - paths.finder.real_finder.Et) <= 2 * self.Et_opt_tol:
            Et = (maxV + minV) / 2
        else:
            Et = (maxV + paths.finder.real_finder.Et - 2*self.Et_opt_tol) / 2
        return Et

    def get_next_mean(self, paths, **kwargs):
        V = paths.get_potential_energy(index=np.s_[1:-1])
        return V.max()#  + (V.max() - V.min())*0.1


    def alternate_energy(self, paths, gptol=None, iter=None):
        if iter % 2 == 0:
            return self.maximum_uncertainty(paths, gptol=gptol)
        E = paths.get_potential_energy(index=np.s_[:])
        self._maximum_energy_checked = True
        self._E_max = E.max()
        return np.argmax(E)

    def check_alternate_energy_convergence(self, paths, idx=None, **kwargs):
        if kwargs['iter'] % 2 == 0:
            return 0
        return self.check_maximum_energy_convergence(paths, idx=idx, **kwargs)

    def alternate_saddle(self, paths, iter=None):
        if iter % 2 == 0:
            return self.maximum_uncertainty(paths)
        E = paths.get_potential_energy(index=np.s_[:])
        X = paths.get_distances(index=np.s_[:])
        dE = np.diff(E)
        dX = np.diif(X)
        ddEdX2 = np.diff(dE / dX) / dX[:-1]
        saddle = np.arange(1, paths.N - 1)[np.abs(ddEdX2) < self.gptol]
        if len(saddle) == 0:
            return np.argmax(E)
        return np.random.choice(saddle)

    def check_alternate_saddle_convergence(self, paths, idx=None, **kwargs):
        if kwargs['iter'] % 2 == 0:
            return False
        blackids = paths.model.data_ids['image']
        E = paths.get_potential_energy(index=np.s_[:])
        data_ids = paths.add_data(index=idx, blackids=blackids)
        imgdata = paths.get_data(data_ids=data_ids)
        if np.abs(E[idx] - imgdata['V'][-1]) < self.gptol:
            return 1
        return 0

    def acquisition(self, paths, phase=None, iter=0):
        if phase is None:
            phase = self.phase
        self._maximum_uncertainty_checked = False
        self._maximum_energy_checked = False
        self._target_energy_checked = False
        phase = self.Phase.lower().replace(" ", "_")
        return getattr(self, phase)(paths, iter=iter)

    def check_convergence(self, paths, phase=None, iter=None, logfile=None,
                          **kwargs):
        phase = phase or self.phase
        imgdata = paths.get_data()
        if len(imgdata['V']) < 3:
            logfile.write("Initial index : %d \n" % (paths.N // 3))
            paths.add_data(index=paths.N // 3)
            logfile.write("Energy added : %.4f\n" % imgdata['V'][-1])
            logfile.flush()
            return False

        idx = self.acquisition(paths, phase=phase, iter=iter)
        logfile.write("Add new idx  : %d \n" % idx)
        logfile.flush()

        phase_name = self.Phase.lower().replace(" ", "_")
        phase = 'check_' + phase_name + '_convergence'
        self.phase += getattr(self, phase)(paths, idx=idx, iter=iter,
                                           logfile=logfile, **kwargs)
        logfile.write("Energy added : %.4f\n" % imgdata['V'][-1])
        logfile.flush()
        if self.phase >= len(self.Phases):
            # logfile.write("Last phase. Checking dS...")
            # if paths.finder.real_finder.isConverged(paths):
            cov_max = np.max(paths.get_covariance())
            logfile.write("Last phase. Checking Cov max %f" % cov_max)
            if cov_max < self.cov_max_tol:
                logfile.write("..Converged!\n")
                return True
            logfile.write(".. Too big not converged\n")
            # dist = frechet_distance(self._prevcoords, paths)
            # logfile.write(" converged, checking displacement %f" % dist)
            # if dist < self.distance_tol:
            #     logfile.write('Distance %f < tol, Converged!' % dist)
            #     return True
            # self._prevcoords = paths.copy()
            return False
            # logfile.write("NOT converged\n")
        # self._prevcoords = paths.copy()
        return False

    def I_prepared_my_paths_in_various_ways(self, paths, **kwargs):
        if 'auto et' == self.Phase.lower():
            pass
        # if True:
            # logfile = kwargs.get('logfile')
            # logfile.write('Prepare my paths in various ways \n')
            # paths.fluctuate(initialize=True)

    def optimize(self, paths, gptol=0.01, maxiter=50, restart=None,
                 **search_kwargs):
        label = getattr(self, 'label', None) or paths.label
        log = self.log or label + '.log'
        logdir = os.path.dirname(log)
        if logdir == '':
            logdir = '.'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        with open(log, 'a+') as logfile:
            dir = os.path.dirname(log)
            if dir == '':
                dir = '.'
            if not os.path.exists(dir):
                os.makedirs(dir)
            if os.path.exists(label + '_pathsdata.db') and self.restart:
                logfile.write("\nReading %s \n" % (label + '_pathsdata.db'))
                data = self._load(paths, filename=label+'_pathsdata.db')
                paths = data['paths']
                iter_number = data['rowid'] + 1
            else:
                filename = label + '_initial'
                logfile.write("Writting %s \n" % (label + '_pathsdata.db'))
                self._save(paths, filename=filename)
                iter_number = 1

            logfile.flush()
            i = iter_number
            while not self.check_convergence(paths, iter=i, logfile=logfile):
                data_ids = paths.model.data_ids['image']
                imgdata = paths.get_data(ids=data_ids)
                dat = [str(d) for d in data_ids]
                logfile.write("Iteration    : %d\n" % i)
                logfile.write("Phase        : %s\n" % self.Phase)
                logfile.write("Number of Dat: %d\n" % len(dat))
                logfile.write("ImgData idx  : %s\n" % ', '.join(dat))
                # logfile.write("Energy added : %.4f\n" % imgdata['V'][-1])
                filename = label + '_{i:02d}'.format(i=i)
                self.I_prepared_my_paths_in_various_ways(paths, logfile=logfile)
                paths.search(real_finder=True, logfile=logfile, **search_kwargs)
                self.results.update(paths.finder.real_finder.results)
                self._save(paths, filename=filename)
                i += 1
                if self.maxtrial < i:
                    logfile.write("Max iteration, %d, reached! \n" % self.maxtrial)
                    break

            dat = [str(d) for d in paths.model.data_ids['image']]
            logfile.write("Iteration    : %d\n" % i)
            logfile.write("Phase        : %s\n" % self.Phase)
            logfile.write("Number of Dat: %d\n" % len(dat))
            logfile.write("ImgData idx  : %s\n" % ', '.join(dat))
            filename = label + '_{i:02d}'.format(i=i)
            self.I_prepared_my_paths_in_various_ways(paths, logfile=logfile)
            paths.search(real_finder=True, logfile=logfile, **search_kwargs)
            self.results.update(paths.finder.real_finder.results)
            self._save(paths, filename=filename)
        return paths

    def _save(self, paths, filename=None, plot_kwargs=None):
        label = getattr(self, 'label', None) or paths.label
        plot_kwargs = plot_kwargs or self.plot_kwargs
        self.plot(paths, filename=filename, **plot_kwargs)
        # paths.plot(filename=filename, savefig=True, gaussian=True)
        pathsdata = PathsData(label + '_pathsdata.db')
        data = [{'paths': paths}]
        pathsdata.write(data=data)

    def _load(self, paths, filename=None, plot_kwargs=None):
        pathsdata = PathsData(filename)
        query = "rowid DESC LIMIT 1;"
        where = " ORDER BY "
        columns = ['rowid', 'paths']
        data = pathsdata.read(query=query, where=where, columns=columns)[0]
        data['paths'] = self.patch_paths(data['paths'])
        #### IDK WHY but remove read-only
        data['paths'].coords = data['paths'].coords.copy()
        return data

    def patch_paths(self, paths):
        if self.__dict__.get('_patch_paths') is not None:
            return self.__dict__.get('_patch_paths')(paths)
        return paths
