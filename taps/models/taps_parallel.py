import os
import select

from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import subprocess

from taps.models import Model


def _std_stream(std, std_queue, kill_sig, poll_obj):
    while not kill_sig.is_set():
        poll_result = poll_obj.poll(0)
        if poll_result:
            std_queue.put(std.stdout.readline())
    kill_sig.clear()
    std.stdout.close()


class Julia(Model):
    """ Externel calculator for parallel calculation

    Parameters
    ----------


    """

    implemented_properties = {'covariance', 'potential', 'gradients',
                              'hessian', 'momentum', 'kinetic_energies',
                              'kinetic_energy_gradient', 'velocity',
                              'acceleration', 'mass', 'effective_mass'}

    def __init__(self, mpi=None, io="SocketIO", io_kwargs={}, **kwargs):

        if type(io) == str:
            from_ = 'taps.io.' + io.lower()
            module = __import__(from_, {}, None, [io])
            self.io = getattr(module, io)(**io_kwargs)
        self.mpi = mpi

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=['potential'], nprc=None,
                  intype=b'2', instruction=b'get_properties', **kwargs):
        nprc = nprc or self.io.nprc

        self.update_coords(coords, nprc)
        # Excute
        self.io.send(properties, intype=intype, instruction=instruction)

        # Write results
        rest = dict([("rank%d" % rank, {}) for rank in range(nprc)])
        returnkey = {
            'all': {'model_kwargs': {'results': None}},
            **rest
            }
        args, resultsdct = self.io.write_parallel(**returnkey)

        for prop in properties:
            resultslist = []
            for i in range(nprc):
                resultslist.append(resultsdct[i]['model_kwargs']['results'][prop])
            # for key, value in resultsdct.items():
            #     resultsorder.append(key)
            #     resultslist.append(value['model_kwargs']['results'][prop])
            self.results[prop] = np.concatenate(resultslist, axis=-1)

    # def get_distances(self, paths, **kwargs):
    #     return self.get_properties(paths, properties='displacements',
    #                                instruction=b'get_distances', **kwargs)

    # def get_momentums(self, paths, **kwargs):
    #     return self.get_properties(paths, properties='momentum',
    #                                instruction=b'get_momentum', **kwargs)

    # def get_kinetic_energies(self, paths, **kwargs):
    #     return self.get_properties(paths, properties='kinetic_energies',
    #                                instruction=b'get_kinetic_energy', **kwargs)

    # def get_kinetic_energy_gradients(self, paths, **kwargs):
    #     return self.get_properties(paths, properties='kinetic_energy_gradient',
    #                                instruction=b'get_kinetic_energy_gradient', **kwargs)

    # def get_velocities(self, paths, **kwargs):
    #     return self.get_properties(paths, properties='velocity',
    #                                instruction=b'get_velocity', **kwargs)

    # def get_accelerations(self, paths, **kwargs):
    #     return self.get_properties(paths, properties='acceleration',
    #                                instruction=b'get_acceleration', **kwargs)

    # def get_accelerations(self, **kwargs):
    #     return self.get_properties(paths, properties='acceleration',
    #                                instruction=b'get_acceleration', **kwargs)

    def get_masses(self, paths, **kwargs):
        return self.get_properties(paths, properties='mass',
                                   instruction=b'get_masses', **kwargs)

    def get_effective_mass(self, paths, **kwargs):
        return self.get_properties(paths, properties='effective_mass',
                                   instruction=b'get_effective_mass', **kwargs)

    def execute_cmd(self, *args, **kwargs):
        """
        --io
            help = "IO mode, `socket`, `file` default is `file`. "
            arg_type = String
            default = "file"
        --host
            help = "another option with an argument"
            # arg_type = String
            # default = "127.0.0.1"
        --port"
            help = "another option with an argument"
            arg_type = Int
            default = 6543
        "--keep_open"
            help = "whether shutdown calculation after finish"
            action = :store_true
        "input_file"
            help = "a positional argument"
            required = false
        """

#        self._udp_queue = Queue()
        self._std_queue = Queue()
#        self._udp_kill_sig = Event()
        self._std_kill_sig = Event()
        self._std_poll = select.poll()

        dirname = os.path.dirname(__file__)
        jldir = "../taps_parallel/io/cmd.jl"
        # modeljl = dirname + '/' + jldir
        modeljl = "/home/schinavro/libCalc/taps/taps_parallel/io/cmd.jl"
        julia = 'julia --project %s ' % modeljl
        argoptions = " ".join(args)
        keyoptions = " ".join(["--%s " % k + str(v).lower() for k, v in kwargs.items()])

        command = julia + argoptions + " " + keyoptions

        if self.mpi is not None:
            command = self.mpi['command'] + " " + command
        print(command)
        pkwargs = {'shell': True, 'universal_newlines': True,
                   'bufsize': 64, 'stdout': subprocess.PIPE,
                   'stderr': subprocess.STDOUT,
                   'preexec_fn': os.setsid}
        self._std = subprocess.Popen(command, **pkwargs)
        self._std_poll.register(self._std.stdout, select.POLLIN)

        self._std_thread = Thread(target=_std_stream,
                                  args=(self._std, self._std_queue,
                                        self._std_kill_sig, self._std_poll))
        self._std_thread.daemon = True
        self._std_thread.start()
        self._std_state = 'open'

    def ping(self):
        self.io.ping()

    def shutdown(self):
        self.io.shutdown()
        self._close_std_thread()

    def initialize(self, *args, **kwargs):
        self.io.read_parallel(*args, **kwargs)

    def initialize_serial(self, *args, **kwargs):
        self.io.read(*args, **kwargs)

    def split_array_dict(self, coords, nprc):
        # Construct coords
        coords_list = self.split_array(coords, nprc)
        rest = dict([('rank%d' % rank,
                     {'coords_kwargs': {'coords': coords_list[rank]}})
                     for rank in range(nprc)])
        return rest

    def split_array(self, coords, nprc):
        return np.array_split(coords, nprc, axis=-1)

    def update_coords(self, coords, nprc):
        rest = self.split_array_dict(coords, nprc)
        coords_dct = {
            # Where to contact
            'all': {},
            **rest
        }
        self.io.read_parallel(**coords_dct)

    def _close_std_thread(self):
        self._std_kill_sig.set()
        self._std_thread.join()

    def _print_stdout(self):
        line = []
        while True:
            try:
                line.append(self._std_queue.get_nowait())
            except Empty:
                break
        # line = self._std_queue.queue
        print("".join(line), end="")
