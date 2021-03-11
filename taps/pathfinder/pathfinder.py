import re
import time
import pickle
import subprocess
import numpy as np
from numpy import newaxis as nax
from numpy.linalg import norm
from collections import OrderedDict
from taps.utils.shortcut import isbool, isdct, isstr, isflt, issclr
from taps.projector import Projector


class PathFinder:
    finder_parameters = {
        'real_finder': {'default': "None", 'assert': 'True'},
        'results': {'default': 'None', 'assert': isdct},
        'relaxed': {'default': 'None', 'assert': isbool},
        'prefix': {'default': 'None', 'assert': isstr},
        'directory': {'default': 'None', 'assert': isstr},
        'prj': {'default': 'Projector()', 'assert': 'True'},
    }

    display_map_parameters = OrderedDict()
    display_graph_parameters = OrderedDict()
    display_graph_title_parameters = OrderedDict()
    relaxed = False

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

    def __setattr__(self, key, value):
        if key[0] == '_':
            super().__setattr__(key, value)
        elif key == 'real_finder':
            if value is None:
                value = eval(self.finder_parameters['real_finder']['default'])
            if isinstance(value, str):
                from_ = 'taps.pathfinder'
                module = __import__(from_, {}, None, [value])
                value = getattr(module, value)()
            super().__setattr__(key, value)
        elif key in self.finder_parameters:
            default = self.finder_parameters[key]['default']
            assertion = self.finder_parameters[key]['assert']
            if value is None:
                value = eval(default.format())
            assert eval(assertion.format(name='value')), (key, value)
            super().__setattr__(key, value)
        elif isinstance(getattr(type(self), key, None), property):
            super().__setattr__(key, value)
        else:
            raise AttributeError('key `%s`not exist!' % key)

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

        if pbs is None:
            new_paths = finder.optimize(paths=paths, **kwargs)
        else:
            new_paths = finder.optimize_pbs(paths=paths, pbs=pbs, **kwargs)

        # if paths is not new_paths:
        #     paths.__dict__.update(new_paths.__dict__)

    def isConverged(self, *args, **kwargs):
        return True

    def optimize_pbs(self, paths=None, pbs=None, **kwargs):
        preset = pbs.get('preset')
        if preset == 'nuri':
            module = "module purge\nmodule load intel/18.0.3 impi/18.0.3\n"
            qsub = ["qsub", "-V", "-N", paths.prefix, "-q", "normal", "-A",
                    'vasp',
                    "-l", "select=1:ncpus=64:mpiprocs=64:ompthreads=1",
                    "-l", self._pbs_walltime]
        if preset == 'fifi':
            module = ''
            qsub = ["qsub", "-N", paths.prefix, "-l", "nodes=1:ppn=5", "-l",
                    self._pbs_walltime]
        self._pbs_write(paths)
        self._qsub(paths, module=module, qsub=qsub)
        if self._write_pbs_only:
            return paths
        new_paths = self._pbs_read(paths)
        return new_paths

    def _pbs_write(self, paths):
        filename = paths.label + 'io'
        result_file = filename + '_result.pkl'
        paths.to_pickle(filename + '.pkl')
        with open(filename + '.py', 'w') as f:
            f.write("import pickle\n")
            f.write("with open('%s', 'rb') as f:\n" % filename)
            f.write("    paths = pickle.load(f)")
            f.write("paths.search(real_finder=True, pbs=None)")
            f.write("with open('%s', 'wb') as f:\n" % result_file)
            f.write("    pickle.dump(f, paths)")

    def _pbs_read(self, paths):
        filename = paths.label + 'io'
        with open(filename + '_result.pkl', 'rb') as f:
            new_paths = pickle.load(f)
        return new_paths

    def _qsub(self, paths, module=None, qsub=None):
        def isRunning(jobNum):
            out = subprocess.Popen('qstat', stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
            qstat = out.communicate()[0].split('\n')
            state = None
            for line in qstat:
                if str(jobNum) in line:
                    state = line.split()[-2]
                    break
            if state == 'R' or state == 'Q':
                return True
            return False
        filename = paths.label
        bash = "#!/bin/bash\n"
        cd = "cd $PBS_O_WORKDIR\n"
        export = "export ASE_VASP_COMMAND='mpirun vasp_gam'\n"
        py = "python %s" % (filename + '.py\n')
        exit = "exit 0"
        echo = ["echo", "$'\n%s'" % (bash + cd + module + export + py + exit)]
        cmd = " ".join(echo) + " | " + " ".join(qsub)
        if self._write_pbs:
            with open(filename + '_qsub.sh', 'w') as f:
                f.write(bash)
                option_flag = False
                for q in qsub:
                    if '-' in q:
                        f.write('\n#PBS ' + q + ' ')
                        option_flag = True
                        continue
                    elif option_flag:
                        f.write(q + ' ')
                        option_flag = False
                        continue
                f.write('\n\n')
                f.write(cd + module + export + py + exit)
        if self._write_pbs_only:
            return None
        print(cmd)
        out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               universal_newlines=True)
        out = out.communicate()[0]
        jobNum = int(re.match(r'(\d+)', out.split()[0]).group(1))
        print('jobNum', jobNum)
        while isRunning(jobNum):
            time.sleep(30)
        return out

    Projector()
