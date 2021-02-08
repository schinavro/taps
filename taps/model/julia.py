import os
import sys
import signal
import pickle
import socket
import asyncio
import time
import json
import numpy as np
import subprocess
from subprocess import TimeoutExpired

import select

from queue import Queue, Empty
from threading import Thread, Event

from taps.model import Model


def _udp_stream(udp, udp_queue, kill_sig):
    while True:
        try:
            udp_queue.put(udp.recv(64000).decode('ascii'))
        except socket.timeout:
            pass
        if kill_sig.is_set():
            kill_sig.clear()
            break


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
        'model_model': {'default': 'None', 'assert': 'True'},
        'model_label': {'default': 'None', 'assert': 'True'},
        'model_potential_unit': {'default': 'None', 'assert': 'True'},
        'model_data_ids': {'default': 'None', 'assert': 'True'},
        'model_kwargs': {'default': 'None', 'assert': 'True'},
        'coords_epoch': {'default': 'None', 'assert': 'True'},
        'coords_unit': {'default': 'None', 'assert': 'True'},
        'finder_finder': {'default': 'None', 'assert': 'True'},
        'finder_prj': {'default': 'None', 'assert': 'True'},
        'finder_label': {'default': 'None', 'assert': 'True'},
        'data_ids': {'default': 'None', 'assert': 'True'},

        'mpi': {'default': 'None', 'assert': 'True'},
        'server': {'default': 'None', 'assert': 'True'}
    }
    implemented_properties = {'covariance', 'potential', 'gradients',
                              'hessian'}

    def __init__(self,
                 model_model=None,
                 model_label=None,
                 model_potential_unit=None,
                 model_data_ids=None,
                 model_kwargs=None,
                 coords_epoch=None,
                 coords_unit=None,
                 finder_finder=None,
                 finder_prj=None,
                 finder_label=None,
                 mpi=None,
                 server=None,
                 debug=False,
                 **kwargs):
        super().model_parameters.update(self.model_parameters)
        self.model_parameters.update(super().model_parameters)

        self.model_model = model_model
        self.model_label = model_label
        self.model_potential_unit = model_potential_unit
        self.model_data_ids = model_data_ids
        self.model_kwargs = model_kwargs
        self.coords_epoch = coords_epoch
        self.coords_unit = coords_unit
        self.finder_finder = finder_finder
        self.finder_prj = finder_prj
        self.finder_label = finder_label
        self.data_ids = {}

        self.mpi = mpi
        self.server = server
        self.debug = debug
        self._server_state = 'close'

        super().__init__(**kwargs)

    def calculate(self, paths, coords, properties=['potential'],
                  model_model=None,
                  model_label=None,
                  model_potential_unit=None,
                  model_data_ids=None,
                  model_kwargs=None,
                  coords_epoch=None,
                  coords_unit=None,
                  finder_finder=None,
                  finder_prj=None,
                  finder_label=None,
                  mpi=None,
                  server=None,
                  debug=None,
                  **kwargs):
        mpi = mpi or self.mpi
        debug = debug or self.debug
        server = server or self.server
        dic = self._generate_input_dict(model_model, model_label,
                                        model_potential_unit,
                                        model_data_ids, model_kwargs,
                                        coords_epoch, coords_unit,
                                        finder_finder, finder_prj,
                                        finder_label)
        if server is None:
            if dic["model_label"][0] == '/':
                filename = dic["model_label"]
            else:
                filename = os.getcwd() + '/' + dic["model_label"]
            results = self._calculate(paths, coords, properties, dic, mpi,
                                      filename, debug, **kwargs)
        else:
            port = server.get('port', 6543)
            instruction = server.get("instruction", b'calculate')
            results = self._calculate_server(paths, coords, properties, dic,
                                             mpi, port, instruction,
                                             debug, **kwargs)

    def _generate_input_dict(self, paths, model_model=None,
                             model_label=None,
                             model_potential_unit=None,
                             model_data_ids=None,
                             model_kwargs=None,
                             coords_epoch=None,
                             coords_unit=None,
                             finder_finder=None,
                             finder_prj=None,
                             finder_label=None):
        dic = {}

        dic["model_model"] = model_model or self.model_model
        assert dic["model_model"] is not None, 'Set model_model'
        dic["model_label"] = model_label or self.model_label or self.label or \
            paths.label or 'julia_model'
        dic["model_potential_unit"] = model_potential_unit or \
            self.model_potential_unit or paths.model.potential_unit
        dic["model_data_ids"] = model_data_ids or self.model_data_ids or \
            self.data_ids
        dic["model_kwargs"] = model_kwargs or self.model_kwargs or kwargs
        dic["coords_epoch"] = coords_epoch or self.coords_epoch or \
            paths.coords.epoch
        dic["coords_unit"] = coords_unit or self.coords_unit or \
            paths.coords.unit
        dic["finder_finder"] = finder_finder or self.finder_finder or \
            paths.finder.__class__.__name__
        dic["finder_prj"] = finder_prj or self.finder_prj or \
            paths.finder.prj.__class__.__name__
        dic["finder_label"] = finder_label or self.finder_label or \
            paths.finder.label or paths.label or 'julia_finder'
        return dic

    def _calculate(self, paths, coords, properties, ikwargs, mpi, filename,
                   debug, **kwargs):
        np.savez(filename + '.npz', coords=coords.T)
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(dic, f)
        modeljl = '/home/schinavro/libCalc/taps/taps_parallel/taps.jl'
        command = 'julia --project %s %s' % (modeljl, filename)
        if mpi is not None:
            command = mpi['command'] + command
        con = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=64)
        while line := con.stdout.readline():
            if debug:
                print(line, end='')
            else:
                time.sleep(5)
        results = {}
        for key, value in np.load('result.npz').items():
            results[key] = value
        return results

    def _open_server(self, mpi, port, debug):

        self._udp_queue = Queue()
        self._std_queue = Queue()
        self._udp_kill_sig = Event()
        self._std_kill_sig = Event()
        self._std_poll = select.poll()

        host = '127.0.0.1'
        modeljl = '/home/schinavro/libCalc/taps/taps_parallel/server.jl'
        command = 'julia --project %s %s' % (modeljl, str(port))
        if mpi is not None:
            command = mpi['command'] + command
        pkwargs = {'shell': True, 'universal_newlines': True,
                   'bufsize': 64, 'stdout': subprocess.PIPE,
                   'stderr': subprocess.STDOUT,
                   'preexec_fn': os.setsid}
        self._std = subprocess.Popen(command, **pkwargs)
        self._std_poll.register(self._std.stdout, select.POLLIN)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.settimeout(1)
        self._udp.bind((host, port + 2))
        self._udp_thread = Thread(target=_udp_stream,
                                  args=(self._udp, self._udp_queue,
                                        self._udp_kill_sig))
        self._std_thread = Thread(target=_std_stream,
                                  args=(self._std, self._std_queue,
                                        self._std_kill_sig, self._std_poll))
        self._udp_thread.daemon = True
        self._std_thread.daemon = True
        self._udp_thread.start()
        self._std_thread.start()

        self._server_state = 'open'
        self._print_udpout()
        self._print_stdout()

    def _close_server(self, port):
        self._print_udpout()
        self._print_stdout()
        host = '127.0.0.1'
        instruction = b'shutdown\n'
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((host, port))
            tcp.sendall(instruction)
            response = tcp.recv(64000)
        if response == b'Shutting down julia\n':
            print("Server succesffuly shutting down")
            time.sleep(2)
        else:
            print("Server fail to shutting down", response)
            # os.killpg(os.getpgid(self._std.pid), signal.SIGINT)
            os.killpg(os.getpgid(self._std.pid), signal.SIGKILL)

    def _close_thread(self):
        self._print_udpout()
        self._print_stdout()
        self._udp_kill_sig.set()
        self._udp_thread.join()
        self._std_kill_sig.set()
        self._std_thread.join()

    def _initialize_server(self, port, *args, **kwargs):
        host = '127.0.0.1'
        instruction = b'standby\n'
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((host, port))
            tcp.sendall(instruction)
            response = tcp.recv(64000)
            if response == b'Roger standby\n':
                input_dict = self._generate_input_dict(*args, **kwargs)
                bytesarr = json.dumps(input_dict).encode("utf-8")
                print("Sending inputs of size %d" % len(bytesarr))
                tcp.send(np.array(len(bytesarr)).tobytes())
                self._print_udpout()
                tcp.sendall(bytesarr)
            else:
                print("Server fail to initialize", response)

        self._print_udpout()
        self._print_stdout()

    def _calculate_server(self, paths, coords, properties, ikwargs, mpi, port,
                          instruction, debug, **kwargs):
        dic["properties"] = properties
        dic["D"] = coords.D
        dic["A"] = coords.A
        dic["N"] = coords.N
        results = {}
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp, \
                socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((host, port))
            udp.bind((host, port + 1))
            udp.settimeout(5)
            s.sendall(instruction)
            while True:
                try:
                    stdline = udp.recv()
                    data = tcp.recv()
                    while True:
                        if debug:
                            print(stdline, end='')

                        if data:
                            results["potential"] = self.deblob(data)
                        # if instruction == b"kill":
                        self._instruction_kill(data, s)
                        if results is not None:
                            break
                finally:
                    tcp.close()
                    udp.close()

        return results

    def _print_udpout(self):
        line = ""
        try:
            line = "".join(list(self._udp_queue.queue))
            self._udp_queue.queue.clear()
        except Empty:
            pass
        print(line, end="")

    def _print_stdout(self):
        line = []
        while True:
            try:
                line.append(self._std_queue.get_nowait())
            except Empty:
                break
        print("".join(line), end="")
