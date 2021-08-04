import os
import select

from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import subprocess

from taps.model import Model


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
    model_parameters = {
        'io': {'default': 'None', 'assert': 'True'},
        'mpi': {'default': 'None', 'assert': 'True'}
    }
    implemented_properties = {'covariance', 'potential', 'gradients',
                              'hessian'}

    def __init__(self, mpi=None, io="SocketIO", io_kwargs={}, **kwargs):
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)

        if type(io) == str:
            from_ = 'taps.io.' + io.lower()
            module = __import__(from_, {}, None, [io])
            self.io = getattr(module, io)(**io_kwargs)
        self.mpi = mpi

        super().__init__(**kwargs)

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
        jldir = "../../taps_parallel/io/cmd.jl"
        modeljl = dirname + '/' + jldir
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

    def initialize(self, *args, **kwargs):
        self.io.read_parallel(*args, **kwargs)

    def initialize_serial(self, *args, **kwargs):
        self.io.read(*args, **kwargs)

    def construct_coords(self, coords):
        # Construct coords
        coords_list = np.array_split(coords, nprc, axis=-1)
        coordskey = {'all': {'host': self.io.chost, "port": self.io.cport},
                     **dict([('%d' % rank, {'coords_kwargs':{
                         'coords': coords_list[rank]}}) for rank in range(nprc)])}
        self.io.update_parallel(**kwargs)

    def calculate(self, paths, coords, properties=['potential'], nprc=None, **kwargs):
        nprc = nprc or self.io.nprc

        self.construct_coords(coords)
        # Excute
        intype, instruction = b'2', b'get_properties'
        self.io.send(intype=intype, instruction=instruction, properties=properties)

        # Write results
        returnkey = {'all': {'model_kwargs': {'results': None}}}
        args, resultsdct = self.io.write_parallel(**returnkey)

        for prop in properties:
            resultslist = []
            for key, value in resultsdct.items():
                resultslist.append(value['model_kwargs']['results'][prop])
            self.results[prop] = np.concatenate(resultslist, axis=-1)

    def shutdown(self):
        self.io.shutdown()
        self._close_std_thread()

    def _close_std_thread(self):
        self._std_kill_sig.set()
        self._std_thread.join()

    def get_mass(self, paths, coords=None, **kwargs):
        return self.calculate(paths, coords, properties=['mass'], **kwargs)

    def get_effective_mass(self, paths, coords=None, **kwargs):
        return self.calculate(paths, coords, properties=['effective_mass'], **kwargs)

    def _print_stdout(self):
        line = []
        while True:
            try:
                line.append(self._std_queue.get_nowait())
            except Empty:
                break
        # line = self._std_queue.queue
        print("".join(line), end="")
