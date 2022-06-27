import os
import numpy as np

from taps.pathfinder import PathFinder
from taps.visualize import view
from taps.db import PathsDatabase


class GPAOPhase:
    def __init__(self, cov_tol=None, data_ids=None):
        self.cov_tol = cov_tol

    def check_convergence(self, paths, printt):
        cov_max = paths.get_covariance().max()
        printt(" Covmax: %.3f" % cov_max)
        if cov_max < self.cov_tol:
            return 1
        printt(" Number of Dat: %d" % len(paths.imgdb.data_ids))
        return 0

    def acquisition(self, paths, printt):
        idx = np.argmax(paths.get_covariance())
        printt(" Paths Index added : %d" % idx)
        ids = paths.add_image_data(index=[idx], force=True)
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
            return paths.add_image_data(index=[idx], force=True)
        E = paths.get_potential()
        return paths.add_image_data(index=[np.argmax(E)], force=True)


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
        paths.model.mean.set_hyperparameters(self.get_next_mean(paths))
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
        return E.min()

    def get_next_mean(self, paths, **kwargs):
        E = paths.get_potential_energy()
        return E.max()


class GPAO(PathFinder):
    """
    from taps.pathfinder import GPAO
    GPAO()
    """

    def __init__(self, real_finder=None, iteration=0,
                 maxtrial=10, logfile=None, plot_kwargs=None,
                 phase_kwargs=None, pathsdatabase=None, regression_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.real_finder = real_finder
        self.iteration = iteration
        self.maxtrial = maxtrial
        self.logfile = logfile
        self.plot_kwargs = plot_kwargs or {}
        self.phase_kwargs = phase_kwargs or {}
        self.pathsdatabase = pathsdatabase or PathsDatabase()
        self.regression_kwargs = regression_kwargs or {}
        # self.pathsdatabase = pathsdatabase

    def optimize(self, paths, label=None, real_finder=None, restart=None,
                 iteration=None, maxtrial=None, logfile=None, plot_kwargs=None,
                 phase_kwargs=None, pathsdatabase=None, **kwargs):
        # Initialize
        label = label or self.label or paths.label
        real_finder = real_finder or self.real_finder
        if restart is not None:
            restart = restart
        self.iteration = iteration or self.iteration
        maxtrial = maxtrial or self.maxtrial
        logfile = logfile or self.logfile
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
                lines = ' '.join([str(lin) for lin in line]) + end
                logfile.write(lines)
                logfile.flush()
            close_log = True
        elif logfile.__class__.__name__ == "TextIOWrapper":
            def printt(*line, end='\n'):
                lines = ' '.join([str(lin) for lin in line]) + end
                logfile.write(lines)
                logfile.flush()
        # Finish Initialize
        # DEFINE for convinience
        # pdb = pathsdatabase
        # End DEFINE

        printt("====================")
        printt("       GPAO")
        printt("====================")

        for name, p_kwargs in phase_kwargs.items():
            class_name = name.replace(" ", "")
            module = __import__('taps.pathfinder.gpao', {}, None, [class_name])
            phase = getattr(module, class_name)(**p_kwargs)
            printt("Phase       : %s" % name)
            while True:
                printt("====================")
                printt("       %s : %03d" % (name, self.iteration))
                printt("====================")
                if maxtrial < self.iteration:
                    printt("Max iteration, %d, reached! " % maxtrial)
                    break
                if phase.check_convergence(paths, printt):
                    printt("Converged! ")
                    break
                new_ids = phase.acquisition(paths, printt)
                self.regression(paths)
                str_ids = [str(id) for id in new_ids]
                printt("Iteration    : %d" % self.iteration)
                printt("imgdb idx  : %s" % ', '.join(str_ids))
                paths.search(real_finder=True, logfile=printt)
                self.results.update(paths.finder.real_finder.results)
                self.save(paths, pathsdatabase=pathsdatabase,
                          plot_kwargs=plot_kwargs, iteration=self.iteration,
                          label=label)
                self.iteration += 1
            printt("")

        if close_log:
            logfile.close()
        return paths

    def regression(self, paths):
        from scipy.optimize import minimize  # , Bounds
        from taps.models.gaussian import Likelihood
        model, database = paths.model, paths.imgdb
        loss_fn = Likelihood(kernel=model.kernel, mean=model.mean,
                             database=database, kernel_prj=model.prj)
        x0 = model.kernel.get_hyperparameters()
        # sigma_f, l^2, sigma_n^e, sigma_n^f
        # bounds = Bounds([1e-2, 1e-2, 1e-5, 1e-6], [5e1, 1e2, 1e-2, 1e-3])
        # sbounds=bounds, method='L-BFGS-B'
        res = minimize(loss_fn, x0, **self.regression_kwargs)
        model.set_lambda(database, Θk=res.x)

    def save(self, paths, pathsdatabase=None, plot_kwargs=None, iteration='',
             label=None):
        pathsdatabase = pathsdatabase or self.pathsdatabase
        plot_kwargs = plot_kwargs or self.plot_kwargs
        iteration = iteration or self.iteration
        label = label or self.label

        if plot_kwargs != {}:
            if iteration != '':
                iteration = '_' + '%03d' % iteration
                plot_kwargs['filename'] = label + iteration
            view(paths, **plot_kwargs)
        pathsdatabase.write(data=[{'paths': paths}])
