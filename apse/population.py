import os
import time
import numpy as np
from random import randrange, random
from ase.pathway.paths import Paths
from ase.pathway.data import PathsData, PickledData, concatenate_atoms_data
from ase.pathway.build import read_csv
from ase.pathway.utils import ImageIndexing
from ase.pathway.distances import FrechetDistance


class Population:
    def __init__(self, pop='population.pkl', all_data='all_data.csv',
                 population_size=20, comparator=None, logfile='csa.log',
                 object='Onsager Machlup'):
        """
        pop : data contains current candidates which frequently changes
        all_data  : every data including unrelaxed coords not change

        """

        if type(pop) == str:
            pop = PickledData(pop)
        elif type(pop) == list:
            candidates = pop
            pop = PickledData('population.pkl')
            pop.write(candidates)
        self._pop = pop
        if type(all_data) == str:
            all_data = PathsData(all_data)
        self.all_data = all_data
        self.object = object
        self.population_size = population_size
        self.dist = FrechetDistance()
        self.logfile = logfile

    @property
    def fitness(self):
        return [self.objective_function(paths) for paths in self.pop[:]]

    def objective_function(self, paths):
        return 1 / paths.results.get(self.object)

    @property
    def metadata(self):
        self._pop.refresh()
        return self._pop.metadata

    @metadata.setter
    def metadata(self, value):
        while True:
            try:
                self._pop.lock.acquire()
                break
            except:
                time.sleep(5)
                continue
        self._pop.refresh()
        self._pop.metadata.update(value)
        self._pop.save()
        self._pop.lock.release()

    @ImageIndexing
    def pop(self, index=np.s_[:]):
        idx = np.arange(self.population_size)[index].reshape(-1)
        self._pop.refresh()
        symbols = self._pop.invariants['symbols']
        pop_invariants = self._pop.invariants['dct']
        pop_variables = self._pop.variables
        variables = {}
        pop = []
        for i in idx:
            for key in self._pop.var_list:
                variables[key] = pop_variables[key][i]
            p = variables.pop('coords')
            pop.append(Paths(symbols, p, **variables, **pop_invariants))
        if len(idx) == 1:
            return pop[0]
        return pop

    @pop.setter
    def pop(self, index=np.s_[:], values=None):
        idx = np.arange(self.population_size)[index].reshape(-1)
        variables = {}
        for key in self._pop.var_list:
            variables[key] = np.array([getattr(p, key) for p in values])
        while True:
            try:
                self._pop.lock.acquire()
                break
            except:
                time.sleep(5)
                continue
        self._pop.refresh()
        for key in self._pop.var_list:
            self._pop.variables[key][idx] = variables[key]
        self._pop.save()
        self._pop.lock.release()

    def get_two_candidates(self):
        pop = self.pop[:]
        if len(pop) < 2:
            self.update()

        if len(pop) < 2:
            return None

        fit = self.fitness
        fmax = max(fit)
        c1 = pop[0]
        c2 = pop[0]
        while c1.id == c2.id:
            nnf = True
            while nnf:
                t = randrange(0, len(pop), 1)
                if fit[t] > random() * fmax:
                    c1 = pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = randrange(0, len(pop), 1)
                if fit[t] > random() * fmax:
                    c2 = pop[t]
                    nnf = False

        return (c1.copy(), c2.copy())

    def crossover(self, m, f, atoms_data_concatenate=False):
        P = np.random.randint(m.P)
        d = m.copy()
        d.coords = np.concatenate([m.coords[..., :P], f.coords[..., P:]], axis=2)
        d.parents = (m.id, f.id)
        d.results = {}
        d.id = 100
        return d

    def mutate(self, d, prob=0.3):
        temperature = np.random.rand()
        if temperature < prob:
            d.fluctuate(temperature=temperature)
        return d

    def get_distances(self, a=None, b=None):
        '''
        a : paths / if None, check for all cand
        b : list of paths / if None, check for all candidates
        '''
        if a is None:
            a = self.pop[:]
        if b is None:
            b = self.pop[:]
        if type(a) == Paths and type(b) == Paths:
            return self.dist(a, b)
        if type(b) == Paths:
            b = [b]
        if type(a) == Paths:
            a = [a]
        return [self.dist(_a, _b) for _a in a for _b in b]

    def dump(self, candidates, mode='w', header=True, index=True,
             **additional_kwargs):
        ids = self.all_data.write(candidates, mode=mode, header=header,
                                  index=index, **additional_kwargs)
        if len(ids) == 1:
            return ids[0]
        return ids

    def __initialize_pop__(self):
        pop = []
        for row in self.all_data.select(query='relaxed==True'):
            s, p, d = row
            pop.append(Paths(s, p, **d))
        self.set_initial_population(pop)

    def set_initial_population(self, pop=None):
        if pop is None:
            pop = []
            for row in self.all_data.select(query='relaxed==True'):
                s, p, d = row
                pop.append(Paths(s, p, **d))
            if len(pop) > self.population_size:
                pop = pop[:self.population_size]
        symbols, _, dct = pop[0].copy(return_dict=True)
        for var in self._pop.var_list:
            self._pop.variables[var] = np.array([getattr(p, var) for p in pop],
                                                dtype=object)
            if dct.get(var) is not None:
                del dct[var]
        self._pop.meta_data['population_size'] = self.population_size
        self._pop.meta_data['dcut'] = np.average(self.get_distances(pop, pop))
        self._pop.invariants['symbols'] = symbols
        self._pop.invariants['dct'] = dct.copy()
        self._pop.save()

    def __add_candidate__(self, paths):
        while True:
            try:
                self._pop.lock.acquire()
                break
            except :
                time.sleep(5)
                continue
        self._pop.refresh()
        pop = self.pop[:]
        dcut = self._pop.meta_data.get('dcut', np.average(self.get_distances()))
        fit = paths.results.get(self.object)
        fitness = self.fitness
        distances = self.get_distances(paths, pop)
        proxyid = np.argmin(distances)
        proxy = np.min(distances)
        if proxy < dcut:
            # check_energy()
            if pop[proxyid].results.get(self.object) < fit:
                # ADD
                self._pop.replace(paths, id=proxyid)
                self._pop.meta_data['dcut'] = dcut / 2
        else:
            theweakest = pop[np.argmin(fitness)]
            if theweakest.results.get(self.object) < fit:
                # ADD
                self._pop.replace(paths, id=proxyid)
        self._pop.save()
        self._pop.lock.release()

    def add_candidate(self, paths):
        while True:
            try:
                self._pop.lock.acquire()
                break
            except Exception as e:
                time.sleep(5)
                continue
        self._pop.refresh()
        pop = self.pop[:]
        dcut = self._pop.meta_data.get('dcut',
                                       np.average(self.get_distances()) / 2)
        fit = self.objective_function(paths)
        fitness = self.fitness
        distances = self.get_distances(paths, pop)
        proxyid = np.argmin(distances)
        proxy = np.min(distances)
        if proxy < dcut:
            # check_energy()
            if fitness[proxyid] < fit:
                # ADD
                self._pop.replace(paths, id=proxyid)
        else:
            theweakest_idx = np.argmin(fitness)
            if fitness[theweakest_idx] < fit:
                # ADD
                self._pop.replace(paths, id=theweakest_idx)
        self._pop.meta_data['dcut'] = dcut * 0.98
        self._pop.save()
        self._pop.lock.release()

    def _write_log(self):
        import time
        cur_time = time.strftime("%b %d %H:%M:%S", time.localtime())
        pop = self.pop[:]
        dcut = self._pop.meta_data['dcut']
        _fit = [paths.results.get(self.object) for paths in pop]
        fit_min = min(_fit)
        fit_ave = np.average(_fit)
        fit_max = max(_fit)

        index_writing = not os.path.isfile(self.logfile)
        with open(self.logfile, 'a') as f:
            if index_writing:
                f.write('       Dcut  fit_min  fit_ave  fit_max' + '\n')
            line = '%.3f %.3f %.3f %.3f' % (dcut, fit_min, fit_ave, fit_max)
            f.write(cur_time + ' : ' + line + '\n')

    def update(self, new_cand=None, initial=False):
        """
        Check if it is valid candidates and then write down to
        candidates file
        """
        if self._pop.meta_data == {}:
            self.__initialize_pop__()
            return

        if new_cand is None and initial is True:
            new_cand = [self.all_data.select(candidates=True)]
        elif type(new_cand) == Paths:
            new_cand = [new_cand]

        for paths in new_cand:
            # self.__add_candidate__(paths)
            self.add_candidate(paths)
        self._write_log()

    def amplify(self, number=20):
        if len(self.candidates) < number:
            for i in range(len(self.candidates), number):
                paths = self.candidates[0].copy()
                self.fluctuate(paths)
                paths.directory += '_%d' % i
                paths.prefix += '_%d' % i
                self.candidates.append(paths)

    def pcr(self, paths, pop_size=20, temperature=0.01, solitary_database=False,
            initial_db_handle=False):
        candidates = []
        for i in range(pop_size):
            _paths = paths.copy()
            _paths.fluctuate(temperature=temperature)
            _paths.directory += '/id_%d' % i
            _paths.prefix += 'id_%d' % i
            _paths.finder.results = {'None yet': 1}
            _paths.finder._pbs = True
            if solitary_database:
                _paths.database = _paths.label
            if initial_db_handle:
                _paths.add_data(index=[0, -1])
            candidates.append(_paths)
        self.dump(candidates, relaxed=False, parents='Orig')

    def periodic_pcr(self, paths, pop_size=20, temperature=0.01, mic_paths=None,
                     solitary_database=False, initial_db_handle=False):
        candidates = []
        for i, mic_p in enumerate(mic_paths):
            for ii in range(pop_size):
                iii = i * pop_size + ii
                _paths = paths.copy()
                _paths.coords = mic_p
                _paths.fluctuate(temperature=temperature)
                _paths.directory += '/id_%d' % iii
                _paths.prefix += 'id_%d' % iii
                _paths.finder.results = {'None yet': 1}
                _paths.finder._pbs = True
                if solitary_database:
                    _paths.database = _paths.label
                if initial_db_handle:
                    _paths.add_data(index=[0, -1])
                candidates.append(_paths)
        self.dump(candidates, relaxed=False, parents='Orig')

    def csa(self, directory=None, subdir=None, name=None,
            solitary_database=False, model_directory=None, model_prefix=None,
            inherit_database=False, maximum_iteration=50, pbs=None,
            write_pbs=None, write_pbs_only=None):
        for i in range(maximum_iteration):
            self.all_data.refresh()
            m, f = self.get_two_candidates()
            d = self.crossover(m, f)
            d = self.mutate(d)
            while True:
                try:
                    self._pop.lock.acquire()
                    break
                except:
                    time.sleep(5)
                    continue
            self.all_data.refresh()
            id = self.dump(d, mode='a', relaxed=False)
            self._pop.lock.release()
            d.directory = directory + '/' + subdir
            d.prefix = 'id_%d/%sid_%d' % (id, name, id)
            if solitary_database:
                d.database = d.label
            if model_directory is not None:
                d.model_directory = model_directory
            if model_prefix is not None:
                d.model_prefix = model_prefix
            if inherit_database:

                d.atomsdata._c = concatenate_atoms_data(d.database,
                                                        m.atomsdata._c,
                                                        f.atomsdata._c)
            d.id = id
            d.finder.tol = 0.01
            d.search()
            # d.finder._pbs = 'fifi'
            # d.finder._write_pbs_only = True
            relaxed_d = d
            while True:
                try:
                    self._pop.lock.acquire()
                    break
                except:
                    time.sleep(5)
                    continue
            self.all_data.refresh()
            id = self.dump(relaxed_d, mode='a', relaxed=True, id=id)
            self._pop.lock.release()
            self.update(relaxed_d)
            self._pop.iteration += 1
            if self._pop.meta_data['dcut'] < 0.05:
                break

    def initial_relax(self, parallel=False, par_num=10, write_pbs_only=False,
                      pbs=False, write_pbs=False):
        def _relax(irow):
            i, row = irow
            time.sleep(((i) % par_num) * 5)
            symbols, coords, dct = row
            paths = Paths(symbols, coords, **dct)
            paths.finder._pbs = pbs
            paths.finder._write_pbs = write_pbs
            paths.finder._write_pbs_only = write_pbs_only
            if os.path.exists(paths.label + 'io_result.csv'):
                paths = read_csv(paths.label + 'io_result.csv', index=-1)
            else:
                print(paths.label)
#                paths.real_finder.search(paths)
                _paths = paths.search()
            return _paths.copy(return_dict=True)
        if not parallel:
            pathsrow = []
            for i, row in enumerate(self.all_data.select()):
                pathsrow.append(_relax((i, row)))
        if parallel:
            from joblib import Parallel, delayed
            pathsrow = Parallel(n_jobs=par_num)(
                delayed(_relax)((i, row)) for i, row in enumerate(
                    self.all_data.select()))
        relaxed_cand = []
        for row in pathsrow:
            s, p, d = row
            relaxed_cand.append(Paths(s, p, **d))
        self.dump(relaxed_cand, mode='w', relaxed=True, parents='Orig')
        self.set_initial_population(relaxed_cand)

    def read_io_results(self, mode='a'):
        relaxed_cand = []
        for i, row in enumerate(self.all_data.select()):
            symbols, coords, dct = row
            paths = Paths(symbols, coords, **dct)
            try:
                cand = read_csv(paths.label + '_result.csv')
            except FileNotFoundError:
                cand = read_csv(paths.label + 'io_result.csv')
            relaxed_cand.append(cand)
        self.dump(relaxed_cand, mode=mode, relaxed=True, parents='Orig')
        self.set_initial_population(relaxed_cand)

    def fluctuate(self, paths, temperature=3):
        """
        First randomly send initial, final position within MIC cell
        Then fluctuate using descrete sine transformation.
        While doing so, it should be kept rotation.
        """
        rcoords = paths.rcoords.copy()
        shape = rcoords.shape
        paths.rcoords = rcoords * temperature * (np.random.random(shape) - 0.5)

    def select(self, **selection):
        """for id in len(pd):
            df at
            df.loc[id]
        _paths1D = self._pd['paths'].values[0]"""
        all_paths = []
        for symbols, coords, dct in self.all_data.select(**selection):
            all_paths.append(Paths(symbols, coords, **dct))
        return all_paths
