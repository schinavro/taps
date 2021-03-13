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

from copy import deepcopy

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
        'model': {'default': 'None', 'assert': 'True'},
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
                 model=None,
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

        self.model = model
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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k[0] == '_':
                continue
            setattr(result, k, deepcopy(v, memo))
        return result

    def __getstate__(self):
        new_dict = {}
        for k, v in self.__dict__.items():
            if k[0] == '_':
                continue
            new_dict[k] = v
        return new_dict

    def calculate(self, paths, coords, properties=['potential'],
                  model=None, model_kwargs=None,
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
        dic = self._generate_input_dict(paths, model, model_kwargs,
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
            results = self._calculate_server(paths, coords, properties, dic,
                                             mpi, debug, **kwargs)
        self.results = results

    def _generate_input_dict(self, paths, model=None,
                             model_kwargs=None,
                             coords_epoch=None,
                             coords_unit=None,
                             finder_finder=None,
                             finder_prj=None,
                             finder_label=None):
        dic = {}
        model_kwargs = model_kwargs or self.model_kwargs
        model_kwargs["label"] = model_kwargs.get("label", self.label)
        model_kwargs["potential_unit"] = model_kwargs.get("potential_unit",
                                                          self.potential_unit)
        model_kwargs["data_ids"] = model_kwargs.get("data_ids" or self.data_ids)
        dic["model"] = model or self.model
        dic["model_kwargs"] = model_kwargs
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

    def _open_server(self, mpi, host=None, port=None, debug=False):
        host = host or self.server['host']
        port = port or self.server['port']

        self._udp_queue = Queue()
        self._std_queue = Queue()
        self._udp_kill_sig = Event()
        self._std_kill_sig = Event()
        self._std_poll = select.poll()

        modeljl = '/home/schinavro/libCalc/taps/taps_parallel/server.jl'
        command = 'julia --project %s %s %s' % (modeljl, host, str(port))
        if mpi is not None:
            command = mpi['command'] + command
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

        while True:
            try:
                self._connect_tcp_server()
                break
            except ConnectionRefusedError:
                time.sleep(1)

    def _connect_tcp_server(self):
        host = self.server['host']
        port = self.server['port']
        instruction = b'standby\n'
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                    tcp.connect((host, port))
                    tcp.sendall(instruction)
                    respond = tcp.recv(64000)
                    if respond == b'Roger standby\n':
                        break
            except ConnectionRefusedError:
                continue

    def _connect_udp_server(self):
        host = self.server['host']
        port = self.server['port'] + 2
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.settimeout(1)
        self._udp.bind((host, port + 2))
        self._udp_thread = Thread(target=_udp_stream,
                                  args=(self._udp, self._udp_queue,
                                        self._udp_kill_sig))
        self._udp_thread.daemon = True
        self._udp_thread.start()
        self._upd_state = 'open'

    def _close_server(self):
        host = self.server['host']
        port = self.server['port']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((host, port))
            instruction = b'shutdown\n'
            tcp.sendall(instruction)
            response = tcp.recv(64000)

        if response == b'Shutting down julia\n':
            print("Server succesffuly shutting down")
            time.sleep(2)
        else:
            print("Server fail to shutting down", response)
            # os.killpg(os.getpgid(self._std.pid), signal.SIGINT)
            os.killpg(os.getpgid(self._std.pid), signal.SIGKILL)

        self._close_std_thread()

    def _close_udp_thread(self):
        self._udp_kill_sig.set()
        self._udp_thread.join()

    def _close_std_thread(self):
        self._std_kill_sig.set()
        self._std_thread.join()

    def _construct_input(self, *args, **kwargs):
        host = self.server['host']
        port = self.server['port']
        instruction = b'construct input\n'
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((host, port))
            tcp.sendall(instruction)
            response = tcp.recv(64000)
            if response == b'Roger standby\n':
                input_dict = self._generate_input_dict(*args, **kwargs)
                bytesarr = json.dumps(input_dict).encode("utf-8")
                tcp.send(np.array(len(bytesarr)).tobytes())
                self._print_udpout()
                tcp.sendall(bytesarr)
            else:
                print("Server fail to initialize", response)

    def _update_model_kwargs(self, *args, **kwargs):
        host = self.server['host']
        port = self.server['port']
        instruction = b'update model_kwargs\n'
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                    tcp.connect((host, port))
                    tcp.sendall(instruction)
                    response = tcp.recv(64000)
                    assert response == b'Roger standby\n', 'Server init failed'
                    input_dict = kwargs
                    bytesarr = json.dumps(input_dict).encode("utf-8")
                    tcp.send(np.array(len(bytesarr)).tobytes())
                    self._print_udpout()
                    tcp.sendall(bytesarr)
                break
            except OSError:
                print("_update can not assign requested address", host, port)
                time.sleep(1)
                continue

    def _construct_coords(self, coords, *args, **kwargs):
        host = self.server['host']
        port = self.server['port']
        instruction = b'construct coords\n'
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                    tcp.settimeout(5)
                    tcp.connect((host, port))
                    tcp.sendall(instruction)
                    response = tcp.recv(64000)
                    assert response == b'Prepare for coords\n'
                    tcp.settimeout(None)
                    DAN = np.array([coords.D, coords.A, coords.N], dtype=np.int64)
                    tcp.sendall(DAN.tobytes())
                    tcp.sendall(coords.tobytes(order='F'))
                break
            except socket.timeout:
                continue
            except OSError:
                print("_coords can not assign requested address", host, port)
                time.sleep(1)
                continue

    def _construct_model(self):
        host = self.server['host']
        port = self.server['port']
        instruction = b'construct model\n'
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((host, port))
            tcp.sendall(instruction)

    def _calculate_server(self, paths, coords, properties, ikwargs, mpi,
                          debug, **kwargs):
        # Short notation
        toint = int.from_bytes
        host = self.server['host']
        port = self.server['port']

        self._construct_coords(coords)
        instruction = b'0' + json.dumps(properties).encode("utf-8")
        results = {}

        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                    tcp.connect((host, port))
                    tcp.sendall(instruction)
                    # i = 0
                    # while i < 10:
                    #    read_sockets, w_s, er_s = select.select([tcp], [],
                    #          [], 1)
                    #    for sock in read_sockets:
                    #        if sock == tcp:
                    #            break
                    #    if debug:
                    #        self._print_stdout()
                    #        self._print_udpout()
                    #    i += 1

                    for i in range(len(properties)):
                        header = toint(tcp.recv(8), 'little', signed=True)
                        meta, count = [], 0
                        n = header
                        while count < n:
                            packet = tcp.recv(n - count)
                            count += len(packet)
                            meta.append(packet)
                        meta = b''.join(meta)
                        rank = toint(meta[:8], 'little', signed=True)
                        shape = np.frombuffer(meta[8:], dtype=np.int64,
                                              count=rank)
                        key = meta[8*rank + 8:n].decode("utf-8")
                        result, count = [], 0
                        n = np.prod(shape) * 8
                        while count < n:
                            packet = tcp.recv(n - count)
                            count += len(packet)
                            result.append(packet)
                        result = b''.join(result)
                        results[key] = np.frombuffer(
                                          result).reshape(np.flip(shape)).T
                break
            except OSError:
                print("_calculate_server can not assign requested address")
                time.sleep(1)
                continue

        return results

    def add_data_ids(self, ids, overlap_handler=True):
        """
        ids : dict of list
        """
        if getattr(self, 'data_ids', None) is None:
            self.data_ids = dict()
        for table_name, id in ids.items():
            if self.data_ids.get(table_name) is None:
                self.data_ids[table_name] = []
            if overlap_handler:
                for i in id:
                    if i not in self.data_ids[table_name]:
                        self.data_ids[table_name].append(int(i))
        self.optimized = False
        self._update_model_kwargs(data_ids=self.data_ids, optimized=False)

    def regression(self):
        self._regression()

    def _regression(self):
        host = self.server['host']
        port = self.server['port']
        instruction = b'model regression\n'
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                    tcp.settimeout(5)
                    tcp.connect((host, port))
                    tcp.sendall(instruction)
                    response = tcp.recv(64000)
                    assert response == b'Prepare for regression\n', response
                    tcp.settimeout(None)
                    response = tcp.recv(64000)
                    assert response == b'Finished regression\n', response
                break
            except socket.timeout:
                continue
            except OSError:
                print("_regress can not assign requested address", host, port)
                time.sleep(1)
                continue

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
        # line = self._std_queue.queue
        print("".join(line), end="")
