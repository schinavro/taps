import os
import time
import numpy as np
from random import randrange, random
from taps.paths import Paths
from taps.db.data import PathsData
from taps.build import read_csv
from taps.utils.utils import ImageIndexing
from taps.distances import FrechetDistance


class OneDict(dict):
    def __init__(self, **kwargs):
        new_kwargs = {}
        new_kwargs.update(kwargs)
        for key, value in kwargs.items():
            new_kwargs[value] = key
        super().__init__(**new_kwargs)


class Population:
    """
    cand = pop.generate_candidate(number=None)
    pop.register_candidate(cand)
    """
    metadata = {'dcut': None, 'iteration': None, }

    def __init__(self, pathsdata='pathsdata.db', number_of_candidates=10,
                 logfile=None):
        self.initialize_pathsdata(pathsdata)
        self.initialize_candidates()
        self.initialize_metadata()
        self.number_of_candidates = number_of_candidates

    def initialize_pathsdata(self, pathsdata):
        key_mapper = OneDict(blb0='init', blb1='fin', int0='candidate',
                             int1='group')
        metadata = {'dcut': 'real', 'iteration': 'int',
                    'current_candidates_ids': 'blob', 'time': 'text'}
        if type(pathsdata) == str:
            pathsdata = PathsData(pathsdata, metadata=metadata,
                                  key_mapper=key_mapper)
        self.pathsdata = pathsdata

    def initialize_metadata(self):
        self.metadata = self.read_metadata()

    def initialize_candidates(self):
        self.candidates = self.read_candidates()

    def write_candidate(self, candidate, trustworthy=1):
        data = {'paths': candidate, 'init': candidate.coords[..., 0],
                'fin': candidate.coords[..., -1], 'candidate': trustworthy}
        id = self.pathsdata.write([data])
        candidate.tag['id'] = id[0]
        return id

    def write_metadata(self, noc=None, iteration=None, dcut=None):
        number_of_candidates = noc or self.count_candidates()
        ids = self.pathsdata.read(query='candidate=1', columns=['rowid'])
        if number_of_candidates < self.number_of_candidates:
            metadata = {'current_candidates_ids': ids, 'time': time.time()}
        else:
            metadata = {'dcut': dcut, 'iteration': iteration,
                        'current_candidates_ids': ids, 'time': time.time()}

        metadata = []
        self.pathsdata.write(data=metadata, table_name='metadata')

    def read_candidates(self, query='candidate=1', columns=['paths', 'rowid']):
        candidates = []
        for datum in self.pathsdata.read(query=query, columns=columns):
            candidate = datum['paths']
            candidate.tag['id'] = datum['rowid']
            candidates.append(candidate)
        return candidates

    def read_similar_candidates(self):
        NotImplementedError()

    def read_metadata(self):
        query = "rowid DESC LIMIT 1;"
        where = " ORDER BY "
        columns = ['dcut', 'iteration', 'current_candidates_ids', 'time']
        metadata = self.pathsdata.read(query=query, columns=columns,
                                       table_name='metadata', where=where)
        if len(metadata) == 0:
            return None
        return metadata[0]

    def update_candidate(self, candidate, trustworthy=1):
        data = {'paths': candidate, 'init': candidate.coords[..., 0],
                'fin': candidate.coords[..., -1], 'candidate': trustworthy}
        id = candidate.tag['id']
        self.pathsdata.update([id], [data])

    def generate_candidate(self, sample=None, p=None):
        """
        Generate candidate based on Paths with given initial and final coord
        if number_of_candidates are bigger than current candidates number,
           generate random paths.
        else, cross over among candidates pool
        samples : list of paths or paths
        """
        number_of_candidates = self.count_candidates()
        if number_of_candidates < self.number_of_candidates:
            assert sample is not None
            if type(sample) is not list:
                sample = [sample]
            ns = len(sample)
            p = p or np.zeros(ns) + 1 / ns
            child = sample[np.random.choice(ns, p=p)].copy()
            child.fluctuate()
            return child
        else:
            mommy, daddy = self.get_two_candidates()
            child = self.crossover(mommy, daddy)
            child = self.mutate(child)
            return child

    def register_candidate(self, candidate=None):
        self.attatch_birth_certificate(candidate)
        candidate.search()
        self.pathsdata.lock()
        number_of_candidates = self.count_candidates()
        print(number_of_candidates)
        if number_of_candidates < self.number_of_candidates:
            self.update_candidate(candidate)
            self.write_metadata(noc=number_of_candidates)
        elif number_of_candidates >= self.number_of_candidates:
            metadata = self.read_metadata()
            iteration = metadata.get('iteration', 0)
            dcut = metadata.get('dcut', np.average(self.get_distances()) / 2)
            fit = self.objective_function(candidate)
            cur_candidates = self.read_candidates()
            # cur_candidates = self.read_similar_candidates()
            fitness = self.objective_function(cur_candidates)
            distances = self.get_distances(candidate, cur_candidates)
            proxyid = np.argmin(distances)
            proxy = np.min(distances)
            if proxy < dcut:
                # check_energy()
                if fitness[proxyid] < fit:
                    # ADD
                    self.update_candidate(candidate, trustworthy=1)
                    self.update_candidate(cur_candidates[proxyid],
                                          trustworthy=0)
            else:
                theweakest_idx = np.argmin(fitness)
                if fitness[theweakest_idx] < fit:
                    # ADD
                    self.update_candidate(cur_candidates[theweakest_idx],
                                          trustworthy=0)
                    self.update_candidate(candidate, trustworthy=1)
            self.write_metadata(noc=number_of_candidates, dcut=dcut * 0.98,
                                iteration=iteration + 1)
        self.pathsdata.release()

    def count_candidates(self):
        return self.pathsdata.count(query='candidate=1')

    @property
    def fitness(self):
        return [self.objective_function(paths) for paths in self.pop[:]]

    def objective_function(self, paths):
        return 1 / paths.results.get(self.object)

    def attatch_birth_certificate(self, candidate):
        id = self.write_candidate(candidate, trustworthy=0)[0]
        prefix = getattr(candidate.finder, 'prefix', None) or candidate.prefix
        tokens = prefix.rsplit('/', 1)
        if len(tokens) == 2:
            sub_dir, prefix = tokens
        else:
            prefix = tokens[0]
        birth_name = 'id_{i:03d}/'.format(i=id) + prefix
        candidate.finder.prefix = birth_name
        self.update_candidate(candidate, trustworthy=0)

    def crossover(self, m, f):
        P = np.random.randint(m.P)
        d = m.copy()
        d.coords = np.concatenate([m.coords[..., :P], f.coords[..., P:]],
                                  axis=2)
        d.tag['parents'] = (m.tag['id'], f.tag['id'])
        d.finder.Et = (m.finder.Et + f.finder.Et) / 2
        data_ids = getattr(d.model, 'data_ids', None)
        if data_ids is None:
            data_ids = {}
        for db_name in data_ids.keys():
            mdb = getattr(m.model, 'data_ids', {}).get(db_name)
            fdb = getattr(f.model, 'data_ids', {}).get(db_name)
            new_db = list(set(mdb).union(set(fdb)))
            print(new_db)
            data_ids[db_name] = new_db
        d.model.data_ids = data_ids
        d.model.optimize = False
        d.reset_cache()
        d.reset_results()
        return d

    def mutate(self, d, prob=0.3):
        d = d.copy()
        temperature = np.random.rand()
        if temperature < prob:
            d.fluctuate()
        return d

    def get_two_candidates(self):
        data = self.pathsdata.read(query='candidate=1', columns=['rowid'])
        ids = []
        for dat_dict in data:
            ids.append(dat_dict['rowid'])
        parents_ids = tuple(np.random.choice(ids, 2, replace=False))
        return self.read_candidates(query='rowid IN (%d, %d)' % parents_ids)

    def get_distances(self, a=None, b=None):
        '''
        a : paths / if None, check for all cand
        b : list of paths / if None, check for all candidates
        '''
        if a is None:
            a = self.read_candidates()
        if b is None:
            b = self.read_candidates()
        if type(a) == Paths and type(b) == Paths:
            return self.dist(a, b)
        if type(b) == Paths:
            b = [b]
        if type(a) == Paths:
            a = [a]
        return [self.dist(_a, _b) for _a in a for _b in b]

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
